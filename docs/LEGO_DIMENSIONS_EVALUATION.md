# Critical evaluation: LEGO stud spacing and dimensions

## Code values (GINN/problems/problem_lego_1xN.py)

| Dimension        | Code value | Unit |
|------------------|------------|------|
| STUD_DIAMETER    | 4.8        | mm   |
| STUD_HEIGHT      | 1.8        | mm   |
| STUD_SPACING     | 8.0        | mm   |
| BRICK_HEIGHT     | 9.6        | mm   |
| BRICK_WIDTH (1x) | 7.8        | mm   |
| brick_length     | n×8.0 − 0.2| mm   |

Stud centers (1xN): `x_i = -brick_length/2 + STUD_SPACING/2 + i * STUD_SPACING` (i = 0..n−1).

---

## Comparison with common LEGO specs

### Stud spacing (center-to-center)

- **Code:** 8.0 mm  
- **Common spec:** 8 mm (standard pitch); finer measurement ~7.986 mm at 25°C.  
- **Verdict:** **Accurate.** 8.0 mm is the standard design value; 7.986 mm is a small tolerance-level difference.

### Brick length (1xN)

- **Code:** `n_studs * 8.0 - 0.2` mm (e.g. 1x2 → 15.8 mm).  
- **Common spec:** 1x2 length 15.8 mm is widely cited.  
- **Verdict:** **Accurate.** Formula and 15.8 mm for 1x2 match common references.

### Stud center formula

- First stud at `-brick_length/2 + 4` mm; then +8 mm per stud.  
- For 1x2 (length 15.8): centers at −3.9 and +4.1 mm → spacing 8.0 mm.  
- **Verdict:** **Correct.** Centers are 8 mm apart; placement is consistent with length 15.8 mm.

### Stud height

- **Code:** 1.8 mm  
- **Common spec:** 1.7 mm (e.g. Brick Owl, Orion Robots; often derived as 9.6/3 − wall thickness).  
- **Verdict:** **Slight inaccuracy.** 1.8 mm is ~6% high; 1.7 mm is the usual cited value.

### Stud diameter

- **Code:** 4.8 mm  
- **Common spec:** 4.8 mm (internal/design); some sources give ~5 mm (outer/tolerance).  
- **Verdict:** **Reasonable.** 4.8 mm matches LEGO unit (3×1.6 mm) and design specs; 5 mm is an outer/tolerance view.

### Brick height (without stud)

- **Code:** 9.6 mm (body); stud adds 1.8 mm → total 11.4 mm.  
- **Common spec:** Brick height 9.6 mm often quoted as total height; stud height 1.7 mm.  
- **Verdict:** **Depends on convention.** If 9.6 mm is “brick only,” code is fine. If 9.6 mm is “brick + stud,” then body would be 7.9 mm. Our 9.6 mm body + 1.7 mm stud = 11.3 mm total is consistent with “tall” brick; correcting stud to 1.7 mm keeps that.

### Brick width (1-wide)

- **Code:** 7.8 mm  
- **Common spec:** 8 − 0.2 mm for base width; some sources say “9.6 mm wide” (different axis or including stud).  
- **Verdict:** **Plausible.** 7.8 mm = 8 − 0.2 is a standard way to get 1-wide base width.

---

## Summary

| Item           | Status   | Action                    |
|----------------|----------|---------------------------|
| Stud spacing   | Accurate | None                      |
| Brick length   | Accurate | None                      |
| Stud centers   | Correct  | None                      |
| Stud height    | ~6% high | Prefer 1.7 mm in code     |
| Stud diameter  | OK       | None                      |
| Brick height   | OK       | None                      |
| Brick width    | OK       | None                      |

**Conclusion:** Stud spacing (8.0 mm) and stud positioning are accurate and consistent with standard LEGO dimensions. The only clear correction is to use **stud height 1.7 mm** instead of 1.8 mm to match common specifications.
