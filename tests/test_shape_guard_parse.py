from mlir_oot.frontend.parse import parse_module


def test_capture_shape_guard_uses_registered_control_flow_assert():
    module = parse_module('''builtin.module {
      func.func @guard(%condition: i1) {
        cf.assert %condition, "captured dimension differs"
        func.return
      }
    }''')
    assert next(op for op in module.walk() if op.name == 'cf.assert').__class__.__name__ == 'AssertOp'
    parse_module(str(module)).verify()
