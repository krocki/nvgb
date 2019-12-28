#include "bootrom.h"
#include "gb_defs.h"
#include <stdio.h>
#include <sys/time.h>
#define CHECK_ERR_CUDA(err) if (err != cudaSuccess) { printf("%s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
//#define DEBUG

double get_time() {
  struct timeval tv; gettimeofday(&tv, NULL);
  return (tv.tv_sec + tv.tv_usec * 1e-6);
}

int read_cart(u8* cart, const char* fname) {
  FILE * file = fopen(fname, "r+");
  if (file == NULL || cart == NULL) return - 1;
  fseek(file, 0, SEEK_END);
  long int size = ftell(file);
  fclose(file);
  file = fopen(fname, "r+");
  int bytes_read = fread(cart, sizeof(u8), size, file);
  printf("read file %s, %d bytes\n", fname, bytes_read);
  return 0; fclose(file);
}
// CPU cycles LUT
__device__ u8 mcycles[256] = {
   4, 12,  8,  8,  4,  4,  8,  4, 20,  8,  8,  8,  4,  4,  8,  4,  // 00-0f
   4, 12,  8,  8,  4,  4,  8,  4, 12,  8,  8,  8,  4,  4,  8,  4,  // 10-1f
  12, 12,  8,  8,  4,  4,  8,  4, 12,  8,  8,  8,  4,  4,  8,  4,  // 20-2f
  12, 12,  8,  8, 12, 12, 12,  4, 12,  8,  8,  8,  4,  4,  8,  4,  // 30-3f
   4,  4,  4,  4,  4,  4,  8,  4,  4,  4,  4,  4,  4,  4,  8,  4,  // 40-4f
   4,  4,  4,  4,  4,  4,  8,  4,  4,  4,  4,  4,  4,  4,  8,  4,  // 50-5f
   4,  4,  4,  4,  4,  4,  8,  4,  4,  4,  4,  4,  4,  4,  8,  4,  // 60-6f
   8,  8,  8,  8,  8,  8,  4,  8,  4,  4,  4,  4,  4,  4,  8,  4,  // 70-7f
   4,  4,  4,  4,  4,  4,  8,  4,  4,  4,  4,  4,  4,  4,  8,  4,  // 80-8f
   4,  4,  4,  4,  4,  4,  8,  4,  4,  4,  4,  4,  4,  4,  8,  4,  // 90-9f
   4,  4,  4,  4,  4,  4,  8,  4,  4,  4,  4,  4,  4,  4,  8,  4,  // a0-af
   4,  4,  4,  4,  4,  4,  8,  4,  4,  4,  4,  4,  4,  4,  8,  4,  // b0-bf
  20, 12, 16, 16, 24, 16,  8, 16, 20, 16, 16,  4, 24, 24,  8, 16,  // c0-cf
  20, 12, 16,  0, 24, 16,  8, 16, 20, 16, 16,  0, 24,  0,  8, 16,  // d0-df
  12, 12,  8,  0,  0, 16,  8, 16, 16,  4, 16,  0,  0,  0,  8, 16,  // e0-ef
  12, 12,  8,  4,  0, 16,  8, 16, 12,  8, 16,  4,  0,  0,  8, 16}; // f0-ff

// 8-bit write
__device__ void w8(gb* g, u16 a, u8 v) {
  switch (a & 0xf000) {
    case 0x8000: // video ram 0x0000-0x0fff
    case 0x9000: // video ram 0x1000-0x1fff
#ifdef DEBUG
      //printf("__w8 VRAM %04x %04x %02x\n", a, a & 0x1fff, v);
#endif
      g->vram[a & 0x1fff] = v;
#ifdef DEBUG
      //printf("%04x %04x %02x\n", a, a & 0x1fff, v);
#endif
      break;
    case 0xc000:
    case 0xd000:
    case 0xe000:
      g->ram[a & 0x1fff] = v; break;
    case 0xf000:
#ifdef DEBUG
      //printf("__w8 HRAM %04x %04x %02x\n", a, a & 0x00ff, v);
#endif
      if (a < 0xfe00) g->ram[a & 0x1fff] = v;
      //if (a < 0xff00) g->oam[a & 0xff] = v;
      //else {
      if (a >= 0xff00) {
        //printf("%04x <- %02x\n", a, v);
        g->hram[a & 0xff] = v;
      }
      //  if (a == 0xff46) { oam_ram(g); }
      break;
    default:
#ifdef DEBUG
      //printf("w8 unhandled %04x\n", a & 0xf000);
#endif
      break;
  }
}
// 16-bit write
__device__ void w16(gb* g, u16 a, u16 v) { w8(g,a,v&0xff); w8(g,a+1,v>>8); }
// 8-bit read
__device__ u8 r8(gb* g, u16 a) {
  if (a <= 0x3fff) return g->rom[a & 0x3fff]; // rom bank 0
  else if (a <= 0x7fff) return g->rom[a & 0x7fff]; // rom bank 1
  else if (a >= 0xc000 && a <= 0xfdff) return g->ram[a & 0x1fff]; // internal ram
  else if (a == 0xff44) return 0x90;
  else if (a >= 0x8000 && a <= 0x9fff) { /*printf("0x%04x = %4x\n", a, g->vram[a & 0x1fff]);*/ return g->vram[a & 0x1fff]; }
  else if (a >= 0xff00 && a <= 0xffff) { return g->hram[a & 0xff]; }
  else {
#ifdef DEBUG
    //printf("r8 a = 0x%04x unhandled\n", a);
#endif
    return 0;
  }
}
// 16-bit read
__device__ u16 r16(gb* g, u16 a) { return ((u16)(r8(g, a+1)) << 8) | (u16)(r8(g, a)); } // read 2 bytes
// operand fetch
__device__ u8 f8(gb *g) { u8 r = r8(g, PC); PC+=1; return r;  } // fetch operand data (byte)
// 16-bit
__device__ u16 f16(gb* g) { u16 r = r16(g, PC); PC+=2; return r; } // fetch operand data (2 bytes)
// 16-bit stack push/pop
__device__ void push16(gb* g, u16 v) { SP -= 2; w16(g, SP, v); } // push onto the stack
__device__ u16 pop16(gb* g) { u16 v = r16(g, SP); SP+=2; return v; } // pop

// opcode handlers
// see: http://www.pastraiser.com/cpu/gameboy/gameboy_opcodes.html for reference
// _TODO: make pretty, i don't like this approach really, but it's easier to debug

// CB extension handler
// OK, this doesn't look nice, it's a hack to get a byte from u16 array with the right endianess
__device__ u8* ptrs(gb* g, u8 idx) {
  u8* _ptrs[8] = {&B, &C, &D, &E, &H, &L, 0, &A}; return _ptrs[idx];
}

#define PTR_REG(x) ptrs(g, (x))
// bitwise ops
// cb
__device__ u8 rlc(gb* g, u8 v) { // rotate left with carry
  u8 c = ((v >> 7) == 0x01); // carry if bit 7 set
  u8 r = (v << 1) | c; // shift and carry previous bit 7 into 0
  fH = 0; fN = 0; fZ = (r == 0); fC = c;
  return r;
}

__device__ u8 rrc(gb* g, u8 v) { // rotate right with carry
  u8 c = (v & 0x01); // carry if bit 0 set
  u8 r = (v >> 1) | (c << 7); // shift and carry previous bit 0 into 7
  fH = 0; fN = 0; fZ = (r == 0); fC = c;
  return r;
}

__device__ u8 rl(gb* g, u8 v) { // rotate left
  u8 c = ((v >> 7) == 0x01); // carry if bit 7 set
  u8 r = (0xff & (v << 1)) | fC;      // shift and carry from flags into 0
  if (r==0) {fZ = 1;} else {fZ = 0;}
  fH = 0; fN = 0; fC = c;
  return r;
}

__device__ u8 rr(gb* g, u8 v) { // rotate right
  u8 c = (v & 0x01); // carry if bit 0 set
  u8 r = (v >> 1) | (fC << 7); // shift and carry from flags into 7
  fH = 0; fN = 0; fZ = (r == 0); fC = c;
  return r;
}

__device__ u8 sla(gb* g, u8 v) { // shift left arithmetic
  u8 c = (v >> 7) & 0x1; // if bit 7 set
  u8 r = (v << 1);
  fH = 0; fN = 0; fZ = (r == 0); fC = c;
  return r;
}

__device__ u8 sra(gb* g, u8 v) { //shift right arithmetic
  u8 c = (v & 0x1); //if bit 0 set
  u8 r = (v >> 1) | (v & 0x80); // shift and extend sign
  fH = 0; fN = 0; fZ = (r == 0); fC = c;
  return r;
}

__device__ u8 srl(gb* g, u8 v) { //shift right logical
  u8 c = (v & 0x1); //if bit 0 set
  u8 r = (v >> 1); // shift
  fH = 0; fN = 0; fZ = (r == 0); fC = c;
  return r;
}

__device__ u8 swap(gb* g, u8 v) {
  fZ = (v==0); fC=0; fN=0; fH=0;
  return ( (v >> 4) | (v << 4) );
}

__device__ void bitchk(gb* g, u8 n, u8 v) {
  u8 r = ((v >> n) & 0x1) == 0;
  //fZ = 0 of bit was 1
  fN = 0; fH = 1; fZ = r;
}

// ops on accumulator (non-CB)
__device__ void rlca(gb* g) { A = rlc(g, A); fZ = 0;}
__device__ void rrca(gb* g) { A = rrc(g, A); fZ = 0;}
__device__ void  rla(gb* g) { A =  rl(g, A); fZ = 0;}
__device__ void  rra(gb* g) { A =  rr(g, A); fZ = 0;}


__device__ u8 inc8(gb* g, u8 v) {
  u8 r = v + 1;
  fZ = (r == 0); fH = ((v & 0x0f) + 1 > 0x0f); fN = 0;
  return r;
}

__device__ u8 dec8(gb* g, u8 v) {
  u8 r = v - 1;
  fZ = (r == 0); fH = ((v & 0x0f) == 0); fN = 1;
  return r;
}

// 8-bit alu ops
__device__ void _add8(gb* g, u8 v, u8 carry) {
  u8 c = carry ? fC : 0;
  u8 r = A;
  r = A + v + c;
  fH = (((A & 0xf) + (v & 0xf) + c) > 0xf) ? 1 : 0;
  fN = 0; fC = (((u16)(A) + (u16)(v) + (u16)(c)) > 0x00ff) ? 1 : 0;
  A = r;
  fZ = (A == 0);
}

__device__ void _sub8(gb * g, u8 v, u8 carry) {
  // use carry?
  u8 c = carry ? fC : 0;
  u8 r = A;
  r = A - v - c;
  // update flags
  fZ = (r == 0); fH = (((A & 0xf) < ((v & 0xf) + c))) ? 1 : 0; fN = 1;
  fC = (((u16)(A) < (u16)(v) + (u16)(c))) ? 1 : 0;
  A = r;
}

__device__ void add8(gb * g, u8 v) { _add8(g, v, 0); }
__device__ void adc8(gb * g, u8 v) { _add8(g, v, 1); }
__device__ void sub8(gb * g, u8 v) { _sub8(g, v, 0); }
__device__ void sbc8(gb * g, u8 v) { _sub8(g, v, 1); }
__device__ void and8(gb * g, u8 v) { A &= v; fZ = (A == 0); fH = 1; fC = 0; fN = 0; }
__device__ void or8 (gb * g, u8 v)  { A |= v; fZ = (A == 0); fH = 0; fC = 0; fN = 0; }
__device__ void xor8(gb * g, u8 v) { A ^= v; fZ = (A == 0); fH = 0; fC = 0; fN = 0; }
__device__ void cp8 (gb * g, u8 v)  { u8 r = A; _sub8(g, v, 0); A = r; }

// 8-bit alu
__device__ void alu(gb *g, u8 op) {
  u8 src_idx = op & 0x7; //last 3 bits are reg#
  u8 src = (((op >> 6) & 0x3) == 3) ? f8(g) : src_idx == 6 ? r8(g, HL) : (*PTR_REG(src_idx));
  u8 n = (op >> 3) & 0x07;

  switch (n) { // subgroup, bits xxNNNyyy
    case 0: add8(g, src); break; // 00000yyy
    case 1: adc8(g, src); break; // 00001yyy
    case 2: sub8(g, src); break; // 00010yyy
    case 3: sbc8(g, src); break; // 00011yyy
    case 4: and8(g, src); break; // 00100yyy
    case 5: xor8(g, src); break; // 00101yyy
    case 6: or8(g, src);  break; // 00110yyy
    case 7: cp8(g, src);  break; // 00111yyy
  };
}

// 8-bit inc/dec
__device__ void incdec(gb* g, u8 op) {
  u8 n = op & 0x3; //dec/inc
  u8 dst_idx = (op >> 3) & 0x7; //last 3 bits are reg#
  u8 src = dst_idx == 6 ? r8(g, HL) : (*PTR_REG(dst_idx));
  src = n ? dec8(g, src) : inc8(g, src);
  if (dst_idx != 6) *PTR_REG(dst_idx) = src; else w8(g, HL, src);
}

// register - register load
__device__ void ldrr(gb* g, u8 op) {
  u8 src_idx = op & 0x7; //last 3 bits are reg#
  u8 dst_idx = (op >> 3) & 0x7; //last 3 bits are reg#
  u8 src = (((op >> 6) & 0x3) == 0) ? f8(g) : src_idx == 6 ? r8(g, HL) : (*PTR_REG(src_idx));
  if (dst_idx != 6) *PTR_REG(dst_idx) = src; else w8(g, HL, src);
}

__device__ void cb_ex(gb* g, u8 x) {

#ifdef DEBUG
  //printf("CB_ex %02x\n", x);
#endif

  u8 src_idx = x & 0x7; //last 3 bits are reg#
  u8 src = src_idx == 6 ? r8(g, HL) : (*PTR_REG(src_idx));
  u8 op_group = (x >> 6) & 0x03;
  u8 n = (x >> 3) & 0x07;
  u8 res = src;

  switch (op_group) {
    case 0:  // opcode == 00xxxyyy
      switch (n) { // subgroup, bits xxNNNyyy
        case 0: res = rlc (g, src);  break; // 00000yyy
        case 1: res = rrc (g, src);  break; // 00001yyy
        case 2: res = rl  (g, src);   break; // 00010yyy
        case 3: res = rr  (g, src);   break; // 00011yyy
        case 4: res = sla (g, src);  break; // 00100yyy
        case 5: res = sra (g, src);  break; // 00101yyy
        case 6: res = swap(g, src); break; // 00110yyy
        case 7: res = srl (g, src);  break; // 00111yyy
      }; break;
    case 1:  // opcode == 01xxxyyy, test bit n
      bitchk(g, n, src); break;
    case 2:  // opcode == 10xxxyyy, clear bit n
      res &= ~(1<<n); break;
    case 3:  // opcode == 11xxxyyy, set bit n
      res |= (1<<n);  break;
  }

  if (src_idx != 6) *PTR_REG(src_idx) = res; else w8(g, HL, res);
  u8 mcycl = (src_idx == 6)  ? 16 : 8; // 16 cycles if hl
  g->cpu_ticks += mcycl;
}

// JUMPS
__device__ void jr  (gb *g) { PC += (s8)(f8(g)); } // jump relative
__device__ void jp  (gb *g) { PC = r16(g, PC); } // jump absolute
__device__ void jphl(gb *g) { PC = HL; }
__device__ void call(gb *g) { push16(g, PC+2); PC=r16(g,PC); } // uncoditional call
__device__ void ret (gb *g) { PC = pop16(g); } // return from call
__device__ void rst (gb *g, u8 v) { push16(g, PC); PC = (u16)(v); } // reset

// 00 - 0f
__device__ void x00(gb *g, u8 op) { /* nop */ }
__device__ void x01(gb *g, u8 op) { BC = r16(g, PC); PC+=2; }
__device__ void x02(gb *g, u8 op) { w8(g,BC,A); }
__device__ void x03(gb *g, u8 op) { BC++; }
__device__ void x04(gb *g, u8 op) { incdec(g, op); }
__device__ void x05(gb *g, u8 op) { incdec(g, op); }
__device__ void x06(gb *g, u8 op) { ldrr(g, op); }
__device__ void x07(gb *g, u8 op) { rlca(g); }
__device__ void x08(gb *g, u8 op) { u16 a = r16(g, PC); w16(g, a, SP); PC+=2; } // LD (a16), SP
__device__ void x09(gb *g, u8 op) { g->unimpl = 1; }
__device__ void x0a(gb *g, u8 op) { A = r8(g,BC); }
__device__ void x0b(gb *g, u8 op) { BC--; }
__device__ void x0c(gb *g, u8 op) { incdec(g, op); }
__device__ void x0d(gb *g, u8 op) { incdec(g, op); }
__device__ void x0e(gb *g, u8 op) { ldrr(g, op);                       } // ld c, imm8
__device__ void x0f(gb *g, u8 op) { rrca(g); }
// 10 - 1f
__device__ void x10(gb *g, u8 op) { g->stopped = 1; }
__device__ void x11(gb *g, u8 op) { DE = r16(g, PC); PC+=2; }
__device__ void x12(gb *g, u8 op) { w8(g,DE,A); }
__device__ void x13(gb *g, u8 op) { DE++; }
__device__ void x14(gb *g, u8 op) { incdec(g, op); }
__device__ void x15(gb *g, u8 op) { incdec(g, op); }
__device__ void x16(gb *g, u8 op) { ldrr(g, op); }
__device__ void x17(gb *g, u8 op) { rla(g); }
__device__ void x18(gb *g, u8 op) { jr(g); } // jr, s8
__device__ void x19(gb *g, u8 op) { g->unimpl = 1; }
__device__ void x1a(gb *g, u8 op) { A = r8(g, DE); }
__device__ void x1b(gb *g, u8 op) { DE--; }
__device__ void x1c(gb *g, u8 op) { incdec(g, op); }
__device__ void x1d(gb *g, u8 op) { incdec(g, op); }
__device__ void x1e(gb *g, u8 op) { ldrr(g, op);                       } // ld e, imm8
__device__ void x1f(gb *g, u8 op) { rra(g); }
// 20 - 2f
__device__ void x20(gb *g, u8 op) { if (!fZ) jr(g); else { PC+=1; }  } // jr nz, s8
__device__ void x21(gb *g, u8 op) { HL = r16(g, PC); PC+=2;          } // ld hl, imm16
__device__ void x22(gb *g, u8 op) { w8(g, HL++, A); }
__device__ void x23(gb *g, u8 op) { HL++; }
__device__ void x24(gb *g, u8 op) { incdec(g, op); }
__device__ void x25(gb *g, u8 op) { incdec(g, op); }
__device__ void x26(gb *g, u8 op) { ldrr(g, op); }
__device__ void x27(gb *g, u8 op) { g->unimpl = 1; }
__device__ void x28(gb *g, u8 op) { if (fZ)  jr(g); else { PC+=1; }   } // jr z, s8
__device__ void x29(gb *g, u8 op) { g->unimpl = 1; }
__device__ void x2a(gb *g, u8 op) { A = r8(g, HL++); }
__device__ void x2b(gb *g, u8 op) { HL--; }
__device__ void x2c(gb *g, u8 op) { incdec(g, op); }
__device__ void x2d(gb *g, u8 op) { incdec(g, op); }
__device__ void x2e(gb *g, u8 op) { ldrr(g, op);                       } // ld l, imm8
__device__ void x2f(gb *g, u8 op) { g->unimpl = 1; }
// 30 - 3f
__device__ void x30(gb *g, u8 op) { if (!fC) jr(g); else { PC+=1; }  } // jr nc,s8
__device__ void x31(gb *g, u8 op) { SP = r16(g, PC); PC+=2;          } // ld sp, imm16
__device__ void x32(gb *g, u8 op) { w8(g, HL--, A); }
__device__ void x33(gb *g, u8 op) { SP++; }
__device__ void x34(gb *g, u8 op) { incdec(g, op); }
__device__ void x35(gb *g, u8 op) { incdec(g, op); }
__device__ void x36(gb *g, u8 op) { ldrr(g, op); }
__device__ void x37(gb *g, u8 op) { g->unimpl = 1; }
__device__ void x38(gb *g, u8 op) { g->unimpl = 1; }
__device__ void x39(gb *g, u8 op) { g->unimpl = 1; }
__device__ void x3a(gb *g, u8 op) { g->unimpl = 1; }
__device__ void x3b(gb *g, u8 op) { SP--; }
__device__ void x3c(gb *g, u8 op) { incdec(g, op); }
__device__ void x3d(gb *g, u8 op) { incdec(g, op); }
__device__ void x3e(gb *g, u8 op) { ldrr(g, op);                       } // ld a, imm8
__device__ void x3f(gb *g, u8 op) { g->unimpl = 1; }

// these are all reg-reg loads
// 40 - 4f
__device__ void x40(gb *g, u8 op) { ldrr(g, op); }; __device__ void x41(gb *g, u8 op) { ldrr(g, op); };
__device__ void x42(gb *g, u8 op) { ldrr(g, op); }; __device__ void x43(gb *g, u8 op) { ldrr(g, op); };
__device__ void x44(gb *g, u8 op) { ldrr(g, op); }; __device__ void x45(gb *g, u8 op) { ldrr(g, op); };
__device__ void x46(gb *g, u8 op) { ldrr(g, op); }; __device__ void x47(gb *g, u8 op) { ldrr(g, op); };
__device__ void x48(gb *g, u8 op) { ldrr(g, op); }; __device__ void x49(gb *g, u8 op) { ldrr(g, op); };
__device__ void x4a(gb *g, u8 op) { ldrr(g, op); }; __device__ void x4b(gb *g, u8 op) { ldrr(g, op); };
__device__ void x4c(gb *g, u8 op) { ldrr(g, op); }; __device__ void x4d(gb *g, u8 op) { ldrr(g, op); };
__device__ void x4e(gb *g, u8 op) { ldrr(g, op); }; __device__ void x4f(gb *g, u8 op) { ldrr(g, op); };
// 50 - 5f
__device__ void x50(gb *g, u8 op) { ldrr(g, op); }; __device__ void x51(gb *g, u8 op) { ldrr(g, op); };
__device__ void x52(gb *g, u8 op) { ldrr(g, op); }; __device__ void x53(gb *g, u8 op) { ldrr(g, op); };
__device__ void x54(gb *g, u8 op) { ldrr(g, op); }; __device__ void x55(gb *g, u8 op) { ldrr(g, op); };
__device__ void x56(gb *g, u8 op) { ldrr(g, op); }; __device__ void x57(gb *g, u8 op) { ldrr(g, op); };
__device__ void x58(gb *g, u8 op) { ldrr(g, op); }; __device__ void x59(gb *g, u8 op) { ldrr(g, op); };
__device__ void x5a(gb *g, u8 op) { ldrr(g, op); }; __device__ void x5b(gb *g, u8 op) { ldrr(g, op); };
__device__ void x5c(gb *g, u8 op) { ldrr(g, op); }; __device__ void x5d(gb *g, u8 op) { ldrr(g, op); };
__device__ void x5e(gb *g, u8 op) { ldrr(g, op); }; __device__ void x5f(gb *g, u8 op) { ldrr(g, op); };
// 60 - 6f
__device__ void x60(gb *g, u8 op) { ldrr(g, op); }; __device__ void x61(gb *g, u8 op) { ldrr(g, op); };
__device__ void x62(gb *g, u8 op) { ldrr(g, op); }; __device__ void x63(gb *g, u8 op) { ldrr(g, op); };
__device__ void x64(gb *g, u8 op) { ldrr(g, op); }; __device__ void x65(gb *g, u8 op) { ldrr(g, op); };
__device__ void x66(gb *g, u8 op) { ldrr(g, op); }; __device__ void x67(gb *g, u8 op) { ldrr(g, op); };
__device__ void x68(gb *g, u8 op) { ldrr(g, op); }; __device__ void x69(gb *g, u8 op) { ldrr(g, op); };
__device__ void x6a(gb *g, u8 op) { ldrr(g, op); }; __device__ void x6b(gb *g, u8 op) { ldrr(g, op); };
__device__ void x6c(gb *g, u8 op) { ldrr(g, op); }; __device__ void x6d(gb *g, u8 op) { ldrr(g, op); };
__device__ void x6e(gb *g, u8 op) { ldrr(g, op); }; __device__ void x6f(gb *g, u8 op) { ldrr(g, op); };
// 70 - 7f
__device__ void x70(gb *g, u8 op) { ldrr(g, op); }; __device__ void x71(gb *g, u8 op) { ldrr(g, op); };
__device__ void x72(gb *g, u8 op) { ldrr(g, op); }; __device__ void x73(gb *g, u8 op) { ldrr(g, op); };
__device__ void x74(gb *g, u8 op) { ldrr(g, op); }; __device__ void x75(gb *g, u8 op) { ldrr(g, op); };
__device__ void x76(gb *g, u8 op) { /* HALT */   }; __device__ void x77(gb *g, u8 op) { ldrr(g, op); };
__device__ void x78(gb *g, u8 op) { ldrr(g, op); }; __device__ void x79(gb *g, u8 op) { ldrr(g, op); };
__device__ void x7a(gb *g, u8 op) { ldrr(g, op); }; __device__ void x7b(gb *g, u8 op) { ldrr(g, op); };
__device__ void x7c(gb *g, u8 op) { ldrr(g, op); }; __device__ void x7d(gb *g, u8 op) { ldrr(g, op); };
__device__ void x7e(gb *g, u8 op) { ldrr(g, op); }; __device__ void x7f(gb *g, u8 op) { ldrr(g, op); };

// reg-reg alu
// 80 - 8f
__device__ void x80(gb *g, u8 op) { alu(g, op);  }; __device__ void x81(gb *g, u8 op) { alu(g, op);  };
__device__ void x82(gb *g, u8 op) { alu(g, op);  }; __device__ void x83(gb *g, u8 op) { alu(g, op);  };
__device__ void x84(gb *g, u8 op) { alu(g, op);  }; __device__ void x85(gb *g, u8 op) { alu(g, op);  };
__device__ void x86(gb *g, u8 op) { alu(g, op);  }; __device__ void x87(gb *g, u8 op) { alu(g, op);  };
__device__ void x88(gb *g, u8 op) { alu(g, op);  }; __device__ void x89(gb *g, u8 op) { alu(g, op);  };
__device__ void x8a(gb *g, u8 op) { alu(g, op);  }; __device__ void x8b(gb *g, u8 op) { alu(g, op);  };
__device__ void x8c(gb *g, u8 op) { alu(g, op);  }; __device__ void x8d(gb *g, u8 op) { alu(g, op);  };
__device__ void x8e(gb *g, u8 op) { alu(g, op);  }; __device__ void x8f(gb *g, u8 op) { alu(g, op);  };
// 90 - 9f;
__device__ void x90(gb *g, u8 op) { alu(g, op);  }; __device__ void x91(gb *g, u8 op) { alu(g, op);  };
__device__ void x92(gb *g, u8 op) { alu(g, op);  }; __device__ void x93(gb *g, u8 op) { alu(g, op);  };
__device__ void x94(gb *g, u8 op) { alu(g, op);  }; __device__ void x95(gb *g, u8 op) { alu(g, op);  };
__device__ void x96(gb *g, u8 op) { alu(g, op);  }; __device__ void x97(gb *g, u8 op) { alu(g, op);  };
__device__ void x98(gb *g, u8 op) { alu(g, op);  }; __device__ void x99(gb *g, u8 op) { alu(g, op);  };
__device__ void x9a(gb *g, u8 op) { alu(g, op);  }; __device__ void x9b(gb *g, u8 op) { alu(g, op);  };
__device__ void x9c(gb *g, u8 op) { alu(g, op);  }; __device__ void x9d(gb *g, u8 op) { alu(g, op);  };
__device__ void x9e(gb *g, u8 op) { alu(g, op);  }; __device__ void x9f(gb *g, u8 op) { alu(g, op);  };
// a0 - af;
__device__ void xa0(gb *g, u8 op) { alu(g, op);  }; __device__ void xa1(gb *g, u8 op) { alu(g, op);  };
__device__ void xa2(gb *g, u8 op) { alu(g, op);  }; __device__ void xa3(gb *g, u8 op) { alu(g, op);  };
__device__ void xa4(gb *g, u8 op) { alu(g, op);  }; __device__ void xa5(gb *g, u8 op) { alu(g, op);  };
__device__ void xa6(gb *g, u8 op) { alu(g, op);  }; __device__ void xa7(gb *g, u8 op) { alu(g, op);  };
__device__ void xa8(gb *g, u8 op) { alu(g, op);  }; __device__ void xa9(gb *g, u8 op) { alu(g, op);  };
__device__ void xaa(gb *g, u8 op) { alu(g, op);  }; __device__ void xab(gb *g, u8 op) { alu(g, op);  };
__device__ void xac(gb *g, u8 op) { alu(g, op);  }; __device__ void xad(gb *g, u8 op) { alu(g, op);  };
__device__ void xae(gb *g, u8 op) { alu(g, op);  }; __device__ void xaf(gb *g, u8 op) { alu(g, op);  };
// b0 - bf;
__device__ void xb0(gb *g, u8 op) { alu(g, op);  }; __device__ void xb1(gb *g, u8 op) { alu(g, op);  };
__device__ void xb2(gb *g, u8 op) { alu(g, op);  }; __device__ void xb3(gb *g, u8 op) { alu(g, op);  };
__device__ void xb4(gb *g, u8 op) { alu(g, op);  }; __device__ void xb5(gb *g, u8 op) { alu(g, op);  };
__device__ void xb6(gb *g, u8 op) { alu(g, op);  }; __device__ void xb7(gb *g, u8 op) { alu(g, op);  };
__device__ void xb8(gb *g, u8 op) { alu(g, op);  }; __device__ void xb9(gb *g, u8 op) { alu(g, op);  };
__device__ void xba(gb *g, u8 op) { alu(g, op);  }; __device__ void xbb(gb *g, u8 op) { alu(g, op);  };
__device__ void xbc(gb *g, u8 op) { alu(g, op);  }; __device__ void xbd(gb *g, u8 op) { alu(g, op);  };
__device__ void xbe(gb *g, u8 op) { alu(g, op);  }; __device__ void xbf(gb *g, u8 op) { alu(g, op);  };
// c0 - cf
__device__ void xc0(gb *g, u8 op) { if (!fZ) ret(g); else g-> cpu_ticks -= 12; } //ret nz
__device__ void xc1(gb *g, u8 op) { BC = pop16(g); }
__device__ void xc2(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xc3(gb *g, u8 op) { jp(g); }
__device__ void xc4(gb *g, u8 op) { if (!fZ) call(g); else { g->cpu_ticks -= 12; PC+=2; } } // call nz, imm16
__device__ void xc5(gb *g, u8 op) { push16(g, BC); }
__device__ void xc6(gb *g, u8 op) { alu(g, op); }
__device__ void xc7(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xc8(gb *g, u8 op) { if (fZ) ret(g); else g->cpu_ticks -= 12; } // ret z
__device__ void xc9(gb *g, u8 op) { ret(g); }
__device__ void xca(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xcb(gb *g, u8 op) { cb_ex(g, f8(g)); } // CB extension
__device__ void xcc(gb *g, u8 op) { if (fZ) call(g); else { g->cpu_ticks -= 12; PC+=2; } } // call z imm16
__device__ void xcd(gb *g, u8 op) { call(g); }
__device__ void xce(gb *g, u8 op) { alu(g, op); }
__device__ void xcf(gb *g, u8 op) { g->unimpl = 1; }
// d0 - df
__device__ void xd0(gb *g, u8 op) { if (!fC) ret(g); else g->cpu_ticks -= 12; } // ret nc
__device__ void xd1(gb *g, u8 op) { DE = pop16(g); }
__device__ void xd2(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xd3(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xd4(gb *g, u8 op) { if (!fC) call(g); else { g->cpu_ticks -= 12; PC+=2; } } // call nc imm 16
__device__ void xd5(gb *g, u8 op) { push16(g,DE); }
__device__ void xd6(gb *g, u8 op) { alu(g, op); }
__device__ void xd7(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xd8(gb *g, u8 op) { if (fC) ret(g); else g-> cpu_ticks -= 12; } // ret c
__device__ void xd9(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xda(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xdb(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xdc(gb *g, u8 op) { if (fC) call(g); else { g->cpu_ticks -= 12; PC+=2; } } // call c imm 16
__device__ void xdd(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xde(gb *g, u8 op) { alu(g, op); }
__device__ void xdf(gb *g, u8 op) { g->unimpl = 1; }
// e0 - ef
__device__ void xe0(gb *g, u8 op) { w8(g, 0xff00 | f8(g), A); }
__device__ void xe1(gb *g, u8 op) { HL = pop16(g); }
__device__ void xe2(gb *g, u8 op) { w8(g, 0xff00 | C, A); }
__device__ void xe3(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xe4(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xe5(gb *g, u8 op) { push16(g,HL); }
__device__ void xe6(gb *g, u8 op) { alu(g, op); }
__device__ void xe7(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xe8(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xe9(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xea(gb *g, u8 op) { w8(g, f16(g), A); }
__device__ void xeb(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xec(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xed(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xee(gb *g, u8 op) { alu(g, op); }
__device__ void xef(gb *g, u8 op) { g->unimpl = 1; }
// f0 - ff
__device__ void xf0(gb *g, u8 op) { A = r8(g, 0xff00 | f8(g));    }
__device__ void xf1(gb *g, u8 op) { AF = pop16(g) & 0xfff0; }
__device__ void xf2(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xf3(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xf4(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xf5(gb *g, u8 op) { push16(g,AF); }
__device__ void xf6(gb *g, u8 op) { alu(g, op);; }
__device__ void xf7(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xf8(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xf9(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xfa(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xfb(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xfc(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xfd(gb *g, u8 op) { g->unimpl = 1; }
__device__ void xfe(gb *g, u8 op) { alu(g, op); }
__device__ void xff(gb *g, u8 op) { g->unimpl = 1; }

// assign ptrs to opcode handlers
// _TODO: make pretty
__device__ void* ops[256] =
{ &x00, &x01, &x02, &x03, &x04, &x05, &x06, &x07, &x08, &x09, &x0a, &x0b, &x0c, &x0d, &x0e, &x0f,
  &x10, &x11, &x12, &x13, &x14, &x15, &x16, &x17, &x18, &x19, &x1a, &x1b, &x1c, &x1d, &x1e, &x1f,
  &x20, &x21, &x22, &x23, &x24, &x25, &x26, &x27, &x28, &x29, &x2a, &x2b, &x2c, &x2d, &x2e, &x2f,
  &x30, &x31, &x32, &x33, &x34, &x35, &x36, &x37, &x38, &x39, &x3a, &x3b, &x3c, &x3d, &x3e, &x3f,
  &x40, &x41, &x42, &x43, &x44, &x45, &x46, &x47, &x48, &x49, &x4a, &x4b, &x4c, &x4d, &x4e, &x4f,
  &x50, &x51, &x52, &x53, &x54, &x55, &x56, &x57, &x58, &x59, &x5a, &x5b, &x5c, &x5d, &x5e, &x5f,
  &x60, &x61, &x62, &x63, &x64, &x65, &x66, &x67, &x68, &x69, &x6a, &x6b, &x6c, &x6d, &x6e, &x6f,
  &x70, &x71, &x72, &x73, &x74, &x75, &x76, &x77, &x78, &x79, &x7a, &x7b, &x7c, &x7d, &x7e, &x7f,
  &x80, &x81, &x82, &x83, &x84, &x85, &x86, &x87, &x88, &x89, &x8a, &x8b, &x8c, &x8d, &x8e, &x8f,
  &x90, &x91, &x92, &x93, &x94, &x95, &x96, &x97, &x98, &x99, &x9a, &x9b, &x9c, &x9d, &x9e, &x9f,
  &xa0, &xa1, &xa2, &xa3, &xa4, &xa5, &xa6, &xa7, &xa8, &xa9, &xaa, &xab, &xac, &xad, &xae, &xaf,
  &xb0, &xb1, &xb2, &xb3, &xb4, &xb5, &xb6, &xb7, &xb8, &xb9, &xba, &xbb, &xbc, &xbd, &xbe, &xbf,
  &xc0, &xc1, &xc2, &xc3, &xc4, &xc5, &xc6, &xc7, &xc8, &xc9, &xca, &xcb, &xcc, &xcd, &xce, &xcf,
  &xd0, &xd1, &xd2, &xd3, &xd4, &xd5, &xd6, &xd7, &xd8, &xd9, &xda, &xdb, &xdc, &xdd, &xde, &xdf,
  &xe0, &xe1, &xe2, &xe3, &xe4, &xe5, &xe6, &xe7, &xe8, &xe9, &xea, &xeb, &xec, &xed, &xee, &xef,
  &xf0, &xf1, &xf2, &xf3, &xf4, &xf5, &xf6, &xf7, &xf8, &xf9, &xfa, &xfb, &xfc, &xfd, &xfe, &xff };

__device__ void ppu_draw_bg(gb *g) {

  u8 gpu_win_on = (REG_LCDC >> 5) & 0x1;//off,on
  u8 lcd_on     = (REG_LCDC >> 7) & 0x1;//off,on

  if (lcd_on) {
    u8 bgy = (REG_SCANLINE+REG_SCY);
    u16 bgtiley = (((u16)bgy) >> 3) & 31;
    s32 winy = gpu_win_on ? (REG_SCANLINE) - (REG_WINY) : -1;
    u16 wintiley = (((u16)winy) >> 3) & 31;
    for (u8 x=0; x<160; x++) {
      u32 bgx = (u32)REG_SCX + (u32)x;
      s32 winx = -(((s32)REG_WINX) - 7) + x;
      u16 tilemapbase;
      u16 tilex, tiley, pixelx, pixely;
      u8 gpu_bgmap =   (REG_LCDC >> 3) & 0x1;//9800-9bff, 9c00-9fff
      u8 gpu_tilemap = (REG_LCDC >> 4) & 0x1;//8800-97ff, 8000-8fff
      u8 gpu_drawbg = (REG_LCDC >> 0) & 0x1;//off,on
      u16 tilebase = (gpu_tilemap ? 0x8000 : 0x8800);
      if (winx >= 0 && winy >=0 ) { // draw window
        u8 gpu_win_map=(REG_LCDC >> 6) & 0x1;//9800-9bff, 9c00-9fff
        tilemapbase=gpu_win_map ? 0x9c00 : 0x9800;
        tiley=wintiley; tilex=(((u16)winx) >> 3); pixely=((u16)winy) & 0x7; pixelx=((u8)winx) & 0x7;
      } else if (gpu_drawbg) { // draw bg
        tilemapbase=gpu_bgmap ? 0x9c00 : 0x9800;
        tiley=bgtiley; tilex=(((u16)bgx) >> 3) & 31; pixely=((u16)bgy) & 0x7; pixelx=((u8)bgx) & 0x7;
      } else {};

      u8 _tilenr = r8(g,tilemapbase + tiley * 32 + tilex);
      u16 tilenr, tileaddress;
      if (tilebase == 0x8800) {
        int8_t nr_s = (s8)_tilenr; s16 nr_s16 = (s16)nr_s + 128; tilenr = (u16)nr_s16;
      }
      else { tilenr = (u16)_tilenr; }
      tileaddress = tilenr * 16 + tilebase;

      u16 a0 = tileaddress + pixely*2;
      u8 data0 = r8(g,a0);
      u8 data1 = r8(g,a0+1);
      u8 color0_idx = ((data0 >> (7-pixelx)) & 0x1);
      u8 color1_idx = ((data1 >> (7-pixelx)) & 0x1);

      u8 color_idx = color0_idx + color1_idx*2;
      //u8 r,g,b;
      u8 color = (REG_BGRDPAL>>(color_idx*2))&0x3;
      //if (color > 0) printf("pix <= %d, %04x x = %d y = %d\n", color, a0, x, REG_SCANLINE);
      //bgprio[x] = color_idx;
      //if (color == 0) {r=255; g=255; b=255;}
      //if (color == 1) {r=192; g=192; b=192;}
      //if (color == 2) {r=96; g=96; b=96;}
      //if (color == 3) {r=0; g=0; b=0;}

      u32 screen_off = ((u32)REG_SCANLINE)*160 + x;
      if (screen_off < 144*160) {
        g->pix[screen_off] = color;
      }
    }
  } else { } // lcd off

}

__device__ void ppu_draw_sprites(gb *g) { }
__device__ void ppu_renderscan(gb *g) { ppu_draw_bg(g); ppu_draw_sprites(g); }
__device__ void blit(gb *g) { g->frame_no++; }
__device__ void check_interrupt_lyc(gb *g) { }

__device__ void ppu_change_mode(gb* g, u8 new_mode) {

  g->ppu_mode = new_mode;

  //REG_LCDSTAT &= ~(0x3); REG_LCDSTAT |= g->ppu_mode & 0x3;
  //REG_LCDSTAT &= ~(0x4); REG_LCDSTAT |= ((REG_SCANLINE == REG_LYC) << 2);

  //u8 irq = 1;
  //u8 m0e = (REG_LCDSTAT >> 3) & 0x1; //hblank int
  //u8 m1e = (REG_LCDSTAT >> 4) & 0x1; //vblank int
  //u8 m2e = (REG_LCDSTAT >> 5) & 0x1; //oam int
  switch (g->ppu_mode) {
    case 0: //irq &= m0e;
            //if (g->enable_ppu)
            ppu_renderscan(g);
            //gpu_hblanking = 1;
            break;
    case 1: /*irq &= m1e; REG_INTF |= 0x01;*/ break;
    case 2: /*irq &= m2e; oam_ram();*/ break;
    case 3: /*irq = 0;*/ break;
  }

  //if (irq) REG_INTF |= 0x2;
}

__device__ void ppu_step(gb* g, u32 ticks) {

  g->ppu_mode_clk += ticks;

  //printf("scanline %d ppu clks %d mode %d\n", REG_SCANLINE, g->ppu_mode_clk, g->ppu_mode);
  if (g->ppu_mode_clk >= 456) { g->ppu_mode_clk -= 456; REG_SCANLINE = (REG_SCANLINE + 1) % 154; }

  check_interrupt_lyc(g);

  if ((REG_SCANLINE >= 144) && (g->ppu_mode != 1)) { ppu_change_mode(g, 1); blit(g); } // vblank
  else {
    // not vblank
    if (g->ppu_mode_clk <= 80) {
      if (g->ppu_mode != 2) ppu_change_mode(g,2); // oam scan
    } else if (g->ppu_mode_clk <= 252) {
      if (g->ppu_mode != 3) ppu_change_mode(g,3); // render line
    } else {
      if (g->ppu_mode != 0) ppu_change_mode(g,0); // hblank
    }
  }

}
__global__ void exec(gb* _g_regs, u8 *prog, u16* _g_rand, int prog_len, int steps, int num_threads) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; // thread idx

  u8 op = 0;

  if (i < num_threads) {

    gb *g = &_g_regs[i]; // local copy of regs
    g->rom = prog;

    for (int j = 0; j < steps; ++j) {

      op = prog[PC]; PC = (PC + 1) % prog_len;
      ((void(*)(gb*,u8))ops[op])(g,op);
      g->cpu_instr++; u8 cpu_cycles = mcycles[op];
      g->cpu_ticks += cpu_cycles;
      ppu_step(g, cpu_cycles);
        printf("I %08d M %08d OP = %02x BC = %04x DE = %04x HL = %04x AF = %04x SP = %04x PC = %04x\n", g->cpu_instr, g->cpu_ticks, op, BC, DE, HL, AF, SP, PC);
      if (g->unimpl) {
        //printf("!!!!    %05d [%04d, %05d] j %5d, ins %08d tic %08d, op %02x BC = %04x DE = %04x HL = %04x AF = %04x SP = %04x PC = %04x ff25 %02x\n",i, blockIdx.x, threadIdx.x, j, g->cpu_instr, g->cpu_ticks, op, BC, DE, HL, AF, SP, PC, g->hram[0x11]); return;
      }
      if (j == steps-1) {
        //printf("last    %05d [%04d, %05d] j %5d, ins %08d tic %08d, op %02x BC = %04x DE = %04x HL = %04x AF = %04x SP = %04x PC = %04x ff25 %02x\n",i, blockIdx.x, threadIdx.x, j, g->cpu_instr, g->cpu_ticks, op, BC, DE, HL, AF, SP, PC, g->hram[0x11]);
      }
      //if (PC == 0x00fe) {
      //  // print mid screen
      //  for (int row=62; row < 82; row++) { printf("%3d: ", row); for (int i = 0; i < 160; i++) { printf("%c", g->pix[160*row + i] > 0 ? '#' : '.'); if (i==159) printf("\n"); } } }
      //if (g->unimpl == 1 || PC >= 0x00fe) { return; };
    }
  }
}

int main(int argc, char **argv) {

  cudaError_t err = cudaSuccess; // for checking CUDA errors

  // Print the vector length to be used, and compute its size
  int num_blocks = 1; int threads_per_block = 1; int prog_len = 32768; int iters = 4; int steps = 1024;

  // override defaults
  if (argc >= 2) num_blocks = atoi(argv[1]);
  if (argc >= 3) threads_per_block = atoi(argv[2]);
  if (argc >= 4) iters = atoi(argv[3]);
  if (argc >= 5) steps = atoi(argv[4]);

  int num_threads = num_blocks * threads_per_block;

  // allocate mem for prog
  u8    *h_prog  = (u8 *)   malloc(prog_len * sizeof(u8));
  const char* fname = "./06-ldrr.gb";
  if (read_cart(h_prog, fname) < 0) { printf("could read %s\n", fname); free(h_prog); return; }

  printf("  main: running %d blocks * %d threads (%d threads total)\n", num_blocks, threads_per_block, num_threads);

  // allocate gb registers / state
  gb    *h_in_regs   = (gb *) malloc(num_threads * sizeof(gb));
  gb    *h_out_regs  = (gb *) malloc(num_threads * sizeof(gb));
  u16   *h_rand      = (u16*) malloc(num_threads * sizeof(u16));

  if (h_rand == NULL || h_in_regs == NULL || h_out_regs == NULL || h_prog == NULL) { fprintf(stderr, "Failed to allocate host mem!\n"); exit(-1); }

  srand(get_time());
  for (int i = 0; i < num_threads; ++i) { h_rand[i] = rand(); }

  // alloc gpu mem
  printf("  main: allocating %zu device bytes\n", prog_len * sizeof(u8) + num_threads * sizeof(gb));

  u8 *d_prog = NULL; gb* d_regs = NULL; u16* d_rand = NULL;
  err = cudaMalloc((void **)&d_prog, prog_len    * sizeof(u8) ); CHECK_ERR_CUDA(err);
  err = cudaMalloc((void **)&d_regs, num_threads * sizeof(gb) ); CHECK_ERR_CUDA(err);
  err = cudaMalloc((void **)&d_rand, num_threads * sizeof(u16)); CHECK_ERR_CUDA(err);

  printf("  main: copying host -> device\n");
  err = cudaMemcpy(d_prog, h_prog,    sizeof(u8 ) * prog_len,    cudaMemcpyHostToDevice);  CHECK_ERR_CUDA(err);
  err = cudaMemcpy(d_regs, h_in_regs, sizeof(gb ) * num_threads, cudaMemcpyHostToDevice);  CHECK_ERR_CUDA(err);
  err = cudaMemcpy(d_rand, h_rand,    sizeof(u16) * num_threads, cudaMemcpyHostToDevice);  CHECK_ERR_CUDA(err);

  for (int j = 0; j < iters; j++ ) {

    //printf("  main: %4d running kernel\n", j);
    cudaDeviceSynchronize();
    double start_time = get_time();
    exec<<<num_blocks, threads_per_block>>>(d_regs, d_prog, d_rand, prog_len, steps, num_threads);
    cudaDeviceSynchronize();

    double walltime = get_time() - start_time;
    err = cudaGetLastError(); CHECK_ERR_CUDA(err);
    //printf("  main: kernel time = %.6f s, %2.6f us/step, %5.3f MHz\n", walltime, 1e6 * (walltime/(steps * num_threads)), ((steps * num_threads)/walltime)/1e6);
  }

  printf("  main: copying device -> host\n");
  err = cudaMemcpy(h_out_regs, d_regs, sizeof(gb) * num_threads, cudaMemcpyDeviceToHost); CHECK_ERR_CUDA(err);
  printf("  main: freeing memory\n");

  // free gpu mem
  err = cudaFree(d_prog); CHECK_ERR_CUDA(err);
  err = cudaFree(d_regs); CHECK_ERR_CUDA(err);
  err = cudaFree(d_rand); CHECK_ERR_CUDA(err);

  // free host mem
  free(h_in_regs); free(h_out_regs); free(h_prog); free(h_rand);

  printf("  main: done.\n");

  return 0;
}
