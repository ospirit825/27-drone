#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <utility>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}
inline volatile sig_atomic_t running = 1;

class LatestFrames {
    cv::VideoCapture capture;
    std::thread worker;
    std::atomic<bool> stop{false};
    std::mutex mutex;
    std::condition_variable changed;
    cv::Mat latest;
    uint64_t sequence = 0;
    Clock::time_point stamp;
    std::string error;
public:
    explicit LatestFrames(const std::string& source) {
        bool opened;
        if (source.rfind("rtsp://", 0) == 0)
            opened = capture.open(source, cv::CAP_FFMPEG,
                {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv::CAP_PROP_READ_TIMEOUT_MSEC, 3000});
        else if (!source.empty() && source.find_first_not_of("0123456789") == std::string::npos)
            opened = capture.open(std::stoi(source));
        else opened = capture.open(source);
        if (!opened) throw std::runtime_error("Cannot open camera/stream: " + source);
        worker = std::thread([this] {
            try {
                while (!stop) {
                    cv::Mat frame;
                    if (!capture.read(frame)) throw std::runtime_error("Camera read failed or stream ended");
                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        latest = std::move(frame);
                        ++sequence;
                        stamp = Clock::now();
                    }
                    changed.notify_one();
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> guard(mutex);
                error = e.what();
            }
            capture.release();
            changed.notify_one();
        });
    }
    cv::Mat next(uint64_t& previous, Clock::time_point& captured, double timeout = 5) {
        std::unique_lock<std::mutex> lock(mutex);
        auto end = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout));
        while (sequence <= previous && error.empty() && running && !stop) {
            if (Clock::now() >= end) throw std::runtime_error("No fresh camera frame within timeout");
            changed.wait_until(lock, std::min(end, Clock::now() + std::chrono::milliseconds(100)));
        }
        if (!running) return {};
        if (sequence <= previous) throw std::runtime_error(error.empty() ? "Camera closed" : error);
        previous = sequence;
        captured = stamp;
        return latest;
    }
    ~LatestFrames() {
        stop = true;
        changed.notify_all();
        if (worker.joinable()) worker.join();
    }
};

class CameraProcess {
    pid_t pid = -1;
    int lockfd = -1;
    static bool portReady() {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("socket failed");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(8554);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        bool ok = connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
        close(fd);
        return ok;
    }
    bool exited() {
        if (pid < 0) return true;
        int status;
        if (waitpid(pid, &status, WNOHANG) == pid) { pid = -1; return true; }
        return false;
    }
public:
    void start(const std::string& log) {
        std::ifstream config("/mnt/data/sensor_cfg.ini");
        std::string text((std::istreambuf_iterator<char>(config)), {});
        if (access("/mnt/system/usr/bin/test_mmf", X_OK) ||
            access("/mnt/system/usr/lib/libsns_gc4653.so", R_OK) ||
            text.find("GCORE_GC4653") == std::string::npos)
            throw std::runtime_error("GC4653 environment incomplete");
        lockfd = open("/tmp/armor_gc4653.lock", O_CREAT | O_RDWR | O_CLOEXEC, 0666);
        if (lockfd < 0 || flock(lockfd, LOCK_EX | LOCK_NB))
            throw std::runtime_error("Cannot lock GC4653; another camera process may be running");
        if (portReady()) throw std::runtime_error("Port 8554 occupied; use --source or stop its server");
        int fd = open(log.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0666);
        if (fd < 0) throw std::runtime_error("Cannot open camera log: " + log);
        std::string libs = "/mnt/system/usr/lib:/usr/bin/lib:";
        if (const char* old = getenv("LD_LIBRARY_PATH")) libs += old;
        pid = fork();
        if (pid == 0) {
            dup2(fd, 1); dup2(fd, 2); close(fd);
            setenv("LD_LIBRARY_PATH", libs.c_str(), 1);
            execl("/mnt/system/usr/bin/test_mmf", "test_mmf", "10", nullptr);
            _exit(127);
        }
        close(fd);
        if (pid < 0) throw std::runtime_error("fork failed");
    }
    std::unique_ptr<LatestFrames> waitForFrames() {
        auto end = Clock::now() + std::chrono::seconds(25);
        std::string lastError;
        while (running && Clock::now() < end) {
            if (exited()) throw std::runtime_error("MMF exited during startup");
            if (portReady()) {
                try {
                    auto frames = std::make_unique<LatestFrames>("rtsp://127.0.0.1:8554/live");
                    uint64_t seq = 0; Clock::time_point stamp;
                    frames->next(seq, stamp, 3);
                    return frames;
                } catch (const std::exception& e) { lastError = e.what(); }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (!running) return {};
        throw std::runtime_error("GC4653 startup timeout: " + lastError);
    }
    ~CameraProcess() {
        for (auto step : {std::pair<int,int>{SIGINT, 8000}, {SIGTERM, 3000}, {SIGKILL, 3000}}) {
            if (exited()) break;
            kill(pid, step.first);
            auto end = Clock::now() + std::chrono::milliseconds(step.second);
            while (!exited() && Clock::now() < end)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (lockfd >= 0) close(lockfd);
    }
};
