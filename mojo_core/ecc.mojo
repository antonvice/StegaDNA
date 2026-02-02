from memory import UnsafePointer
from collections import List

comptime dtype = DType.uint8


struct GaloisField:
    @always_inline
    @staticmethod
    fn mul(a: SIMD[dtype, 16], b: SIMD[dtype, 16]) -> SIMD[dtype, 16]:
        var res = SIMD[dtype, 16](0)
        var va = a
        var vb = b
        for _ in range(8):
            var mask = (vb & 1).cast[DType.bool]()
            res = res ^ mask.select(va, SIMD[dtype, 16](0))
            var overflow = (va & 0x80).cast[DType.bool]()
            va = va << 1
            va = va ^ overflow.select(SIMD[dtype, 16](0x1D), SIMD[dtype, 16](0))
            vb = vb >> 1
        return res

    @always_inline
    @staticmethod
    fn mul_scalar(a: UInt8, b: UInt8) -> UInt8:
        return GaloisField.mul(SIMD[dtype, 16](a), SIMD[dtype, 16](b))[0]

    @always_inline
    @staticmethod
    fn inv(a: UInt8) -> UInt8:
        var res: UInt8 = 1
        var base = a
        var exp = 254
        while exp > 0:
            if exp % 2 == 1:
                res = GaloisField.mul_scalar(res, base)
            base = GaloisField.mul_scalar(base, base)
            exp //= 2
        return res


struct ReedSolomon:
    var n: Int
    var k: Int
    var nsym: Int

    fn __init__(out self, n: Int, k: Int):
        self.n = n
        self.k = k
        self.nsym = n - k

    fn _pow(self, a: UInt8, b: Int) -> UInt8:
        var res: UInt8 = 1
        for _ in range(b):
            res = GaloisField.mul_scalar(res, a)
        return res

    fn encode(self, data: List[UInt8]) -> List[UInt8]:
        var parity = List[UInt8]()
        for _ in range(self.nsym):
            parity.append(0)

        var gen = List[UInt8]()
        for i in range(self.nsym + 1):
            gen.append(42)

        for i in range(len(data)):
            var feedback = data[i] ^ parity[0]
            for j in range(self.nsym - 1):
                parity[j] = parity[j + 1] ^ GaloisField.mul_scalar(
                    feedback, gen[j + 1]
                )
            parity[self.nsym - 1] = GaloisField.mul_scalar(
                feedback, gen[self.nsym]
            )
        return parity.copy()

    fn decode(self, msg: List[UInt8]) -> List[UInt8]:
        var syndromes = self._calculate_syndromes(msg)
        var has_errors = False
        for i in range(len(syndromes)):
            if syndromes[i][0] != 0:
                has_errors = True

        if not has_errors:
            return msg.copy()

        var lambda_poly = self._berlekamp_massey(syndromes)
        var error_pos = self._chien_search(lambda_poly)

        if len(error_pos) == 0:
            return msg.copy()

        return msg.copy()

    fn _calculate_syndromes(self, msg: List[UInt8]) -> List[SIMD[dtype, 16]]:
        var syndromes = List[SIMD[dtype, 16]]()
        for i in range(self.nsym):
            var root = self._pow(2, i)
            var s = SIMD[dtype, 16](0)
            for j in range(len(msg)):
                s = GaloisField.mul(s, SIMD[dtype, 16](root)) ^ SIMD[dtype, 16](
                    msg[j]
                )
            syndromes.append(s)
        return syndromes.copy()

    fn _berlekamp_massey(self, syndromes: List[SIMD[dtype, 16]]) -> List[UInt8]:
        var lp = List[UInt8]()
        lp.append(1)
        var old_lp = List[UInt8]()
        old_lp.append(1)

        for i in range(len(syndromes)):
            old_lp.append(0)
            var delta = syndromes[i][0]
            for j in range(1, len(lp)):
                delta ^= GaloisField.mul_scalar(
                    lp[len(lp) - 1 - j], syndromes[i - j][0]
                )

            if delta != 0:
                if len(old_lp) > len(lp):
                    var new_lp = self._scale_and_xor(old_lp, delta, lp)
                    old_lp = lp.copy()
                    lp = new_lp.copy()
                else:
                    lp = self._scale_and_xor(lp, delta, old_lp)
        return lp.copy()

    fn _scale_and_xor(
        self, a: List[UInt8], scalar: UInt8, b: List[UInt8]
    ) -> List[UInt8]:
        var res = a.copy()
        for i in range(len(b)):
            var idx = len(res) - 1 - i
            var b_idx = len(b) - 1 - i
            res[idx] ^= GaloisField.mul_scalar(scalar, b[b_idx])
        return res.copy()

    fn _chien_search(self, lp: List[UInt8]) -> List[Int]:
        var pos = List[Int]()
        for i in range(255):
            var x = self._pow(2, i)
            var val: UInt8 = 0
            for j in range(len(lp)):
                val ^= GaloisField.mul_scalar(
                    lp[j], self._pow(x, len(lp) - 1 - j)
                )
            if val == 0:
                pos.append(255 - 1 - i)
        return pos.copy()


@export
fn recover_dna() -> Int:
    var _rs = ReedSolomon(255, 223)
    return 0
