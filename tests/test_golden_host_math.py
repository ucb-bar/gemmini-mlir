import unittest
from mlir_oot.frontend.parse import parse_module
from mlir_oot.golden_host_math import rewrite_roundeven, SYMBOL


class HostMathTest(unittest.TestCase):
    def test_only_binary32_roundeven_is_routed(self):
        module = parse_module('''builtin.module {
          func.func @f(%a: f32, %b: f64) -> (f32, f64) {
            %x = math.roundeven %a : f32
            %y = math.roundeven %b : f64
            func.return %x, %y : f32, f64
          }
        }''')
        self.assertEqual(rewrite_roundeven(module), 1)
        module.verify()
        self.assertIn('func.call @' + SYMBOL, str(module))
        self.assertEqual(str(module).count('math.roundeven'), 1)
        self.assertEqual(rewrite_roundeven(module), 0)

    def test_reserved_helper_symbol_refuses_collision(self):
        module = parse_module('''builtin.module {
          func.func private @gemmini_golden_roundevenf(f32) -> f32
          func.func @f(%a: f32) -> f32 {
            %x = math.roundeven %a : f32
            func.return %x : f32
          }
        }''')
        with self.assertRaisesRegex(ValueError, 'symbol'):
            rewrite_roundeven(module)

if __name__ == '__main__':
    unittest.main()
