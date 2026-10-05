import unittest
from mlir_oot.golden_device_profile import emit, parse_profile

class BoundaryProfileTest(unittest.TestCase):
    manifest = {'expected_device_calls':3,'boundaries':[{'id':0},{'id':1}]}
    valid = ('PROFILE_SUM 31 13 18 3 0\nPROFILE_CALL 0 0 10 4\n'
             'PROFILE_CALL 1 1 5 3\nPROFILE_CALL 2 0 2 6\nPROFILE_TAIL 1\n')
    def test_complete_intervals(self):
        self.assertEqual(parse_profile(self.valid,self.manifest)['host_gap_counter'],18)
    def test_missing_boundary_refused(self):
        with self.assertRaisesRegex(ValueError,'cover'):
            parse_profile(self.valid.replace('PROFILE_CALL 1 1','PROFILE_CALL 1 0'),self.manifest)
    def test_nonconserved_intervals_refused(self):
        with self.assertRaisesRegex(ValueError,'conserve'):
            parse_profile(self.valid.replace('PROFILE_TAIL 1','PROFILE_TAIL 2'),self.manifest)
    def test_overflow_refused(self):
        with self.assertRaisesRegex(ValueError,'cover'):
            parse_profile(self.valid.replace('31 13 18 3 0','31 13 18 3 1'),self.manifest)
    def test_unsupported_abi_refused(self):
        with self.assertRaises(ValueError):emit([('unsafe;name',3)])
        with self.assertRaises(ValueError):emit([('valid_name',5)])
