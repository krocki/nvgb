#include <stdint.h>
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;
typedef  int8_t s8; typedef  int16_t s16; typedef  int32_t s32; typedef  int64_t s64;

// struct for keeping the state of the system
typedef struct {

  // CPU regs (96 bits)
  // 8 x  8-bit (A,F,B,C,D,E,H,L)
  // 2 x 16-bit (SP, PC)
  union {
    struct {
      u8 C; u8 B;
      u8 E; u8 D;
      u8 L; u8 H;
      union {
        struct { u8 unused:4; u8 FC:1; u8 FH:1; u8 FN:1; u8 FZ:1;};
        u8 F; }; u8 A;
      u16 SP; u16 PC;
    };
    u16 regs[6];
  };

  // 'cpu' mem
  u8* rom;         // program      0x0000-0x7fff
  u8 ram[0x2000];  // internal ram 0xa000-0xfdff
  u8 vram[0x2000]; // video ram    0x8000-0x9fff
  u8 hram[256];    // i/o+high ram 0xff00-0xffff
  u8 stopped;

  // 'ppu'
  u8 pix[160*144]; // screen: 160x144
  u8 ppu_mode;
  u8 enable_ppu;

  // counters
  u32 cpu_instr;
  u32 cpu_ticks;
  u32 ppu_mode_clk;
  u32 frame_no;

  // extra registers for handling register transfers
  u8 src_reg; u8 dst_reg; u8 unimpl;

} gb;

#define BC (g->regs[0])
#define DE (g->regs[1])
#define HL (g->regs[2])
#define AF (g->regs[3])
#define SP (g->regs[4])
#define PC (g->regs[5])

#define B (g->B)
#define C (g->C)
#define D (g->D)
#define E (g->E)
#define H (g->H)
#define L (g->L)
#define A (g->A)
#define F (g->F)

#define fC (g->FC)
#define fN (g->FN)
#define fZ (g->FZ)
#define fH (g->FH)

// REGS
// serial link
#define REG_SERIAL   (g->hram[0x01])
// timer
#define REG_TIM_DIV  (g->hram[0x04])
#define REG_TIM_TIMA (g->hram[0x05])
#define REG_TIM_TMA  (g->hram[0x06])
#define REG_TIM_TAC  (g->hram[0x07])

// LCD control
#define REG_LCDC     (g->hram[0x40])
// LCD status
#define REG_LCDSTAT  (g->hram[0x41])
// scroll y
#define REG_SCY      (g->hram[0x42])
// scroll y
#define REG_SCX      (g->hram[0x43])
// current line
#define REG_SCANLINE (g->hram[0x44])
//LYC
#define REG_LYC      (g->hram[0x45])
#define REG_OAMDMA   (g->hram[0x46])
//sprite palette
#define REG_OBJPAL0  (g->hram[0x48])
#define REG_OBJPAL1  (g->hram[0x49])
//window coords
#define REG_WINY     (g->hram[0x4a])
#define REG_WINX     (g->hram[0x4b])
//bootrom off
#define REG_BOOTROM  (g->hram[0x50])
// interrupt flags
#define REG_INTF     (g->hram[0x0f])
// interrupt master eng->hram[0x
#define REG_INTE     (g->hram[0xff])
// backgroud paletter
#define REG_BGRDPAL  (g->hram[0x47])
