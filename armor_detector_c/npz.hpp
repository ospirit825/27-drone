#pragma once
#include <zlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// NumPy-compatible, uncompressed ZIP containing one little-endian float32 array.
inline void saveRaw(std::string path, const float* data) {
    if (path.size() < 4 || path.substr(path.size()-4) != ".npz") path += ".npz";
    const std::string name = "output0_Concat_f32.npy";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 36, 8400, 1), }";
    header.append((64 - ((10 + header.size() + 1) % 64)) % 64, ' ');
    header += '\n';
    std::vector<unsigned char> payload(10 + header.size() + 36*8400*4);
    memcpy(payload.data(), "\x93NUMPY\x01\x00", 8);
    payload[8] = header.size() & 255; payload[9] = header.size() >> 8;
    memcpy(payload.data()+10, header.data(), header.size());
    uint16_t endian = 1;
    if (*reinterpret_cast<unsigned char*>(&endian) != 1) throw std::runtime_error("NPZ needs little endian host");
    memcpy(payload.data()+10+header.size(), data, 36*8400*4);
    uint32_t crc = crc32(0, payload.data(), payload.size()), size = payload.size();
    std::ofstream f(path, std::ios::binary);
    auto u16 = [&](uint16_t v) { f.put(v & 255); f.put(v >> 8); };
    auto u32 = [&](uint32_t v) { u16(v & 65535); u16(v >> 16); };
    u32(0x04034b50); u16(20); u16(0); u16(0); u16(0); u16(0);
    u32(crc); u32(size); u32(size); u16(name.size()); u16(0);
    f << name; f.write(reinterpret_cast<char*>(payload.data()), payload.size());
    uint32_t offset = 30 + name.size() + size;
    u32(0x02014b50); u16(20); u16(20); u16(0); u16(0); u16(0); u16(0);
    u32(crc); u32(size); u32(size); u16(name.size()); u16(0); u16(0);
    u16(0); u16(0); u32(0); u32(0); f << name;
    u32(0x06054b50); u16(0); u16(0); u16(1); u16(1);
    u32(46 + name.size()); u32(offset); u16(0);
    f.close();
    if (!f) throw std::runtime_error("Cannot save NPZ: " + path);
}
