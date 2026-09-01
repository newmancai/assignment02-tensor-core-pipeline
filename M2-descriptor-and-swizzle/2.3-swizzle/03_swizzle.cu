// 问题 2.3:三种 swizzle 模式的地址映射。
//
// 输入:元素的逻辑坐标(atom 内行号 row、行内字节偏移 colByte),
// 输出:atom 内的物理字节偏移。三种模式的 atom 尺寸:
//   128B swizzle:8 行 x 128B;64B:8 行(取 4 行一周期)x 64B;
//   32B:8 行(2 行一周期)x 32B。
// 课上 S064 推的是 128B 的地址位异或;64B/32B 的参与位段自己从
// PTX ISA 的 swizzling 一节推。16B 以内的低位永远直通。
//
// 判测(纯 host,无卡):
//   1) 双射:atom 内所有 (row, colByte) 的输出恰好铺满 atom 且无碰撞;
//   2) 列访问无冲突:固定逻辑 16B chunk 下标 j,遍历一个周期内的行,
//      物理 chunk 必须两两不同(descriptor 路径按列取数的要求;
//      单纯 padding 或恒等映射会在这里现形)。
// 运行:make run/m2_smem/03_swizzle
//
// 你的 swizzle_128B 会在 3.2 里直接用于 smem staging——那里是真硬件
// 判测:布局错,GEMM 结果必错。
#include <cstdio>
#include <cstring>

// Swizzling keeps the low 4 address bits intact, so each 16B chunk is the
// indivisible unit.  The row bits XOR the logical chunk index, while the row
// base and the byte position inside the chunk remain unchanged.
static int swizzle_offset(int row, int colByte, int rowBytes,
                          int rowPeriod) {
    const int chunk = colByte >> 4;
    const int byteInChunk = colByte & 0xf;
    const int physicalChunk = chunk ^ (row & (rowPeriod - 1));
    return row * rowBytes + (physicalChunk << 4) + byteInChunk;
}

static int swizzle_128B(int row, int colByte) {
    return swizzle_offset(row, colByte, 128, 8);
}

static int swizzle_64B(int row, int colByte) {
    return swizzle_offset(row, colByte, 64, 4);
}

static int swizzle_32B(int row, int colByte) {
    return swizzle_offset(row, colByte, 32, 2);
}

// 以下为判测,不需要修改。
static int check_mode(const char* name, int (*fn)(int, int), int rowBytes,
                      int period) {
    const int rows = 8;
    int atom = rows * rowBytes;
    static char hit[8 * 128];
    memset(hit, 0, atom);
    int bad = 0;
    // 约定:输出是 atom 内偏移,行基址 row*rowBytes 由映射自己含入。
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < rowBytes; c++) {
            int off = fn(r, c);
            if (off < 0 || off >= atom || hit[off]) bad++;
            else hit[off] = 1;
        }
    if (bad) {
        printf("%s FAIL:双射检查,%d 处越界或碰撞\n", name, bad);
        return 1;
    }
    // 列访问:固定 16B chunk j,一个周期内的行的物理 chunk 应两两不同
    int chunks = rowBytes / 16;
    for (int j = 0; j < chunks; j++) {
        unsigned seen = 0;
        for (int r = 0; r < period; r++) {
            int off = fn(r, j * 16);
            int physChunk = (off % rowBytes) / 16;
            if (seen & (1u << physChunk)) {
                printf("%s FAIL:列 %d 在行 0..%d 内物理 chunk 重复\n", name,
                       j, period - 1);
                return 1;
            }
            seen |= 1u << physChunk;
        }
    }
    printf("%s PASS\n", name);
    return 0;
}

int main() {
    int bad = 0;
    bad += check_mode("128B", swizzle_128B, 128, 8);
    bad += check_mode(" 64B", swizzle_64B, 64, 4);
    bad += check_mode(" 32B", swizzle_32B, 32, 2);
    return bad != 0;
}
