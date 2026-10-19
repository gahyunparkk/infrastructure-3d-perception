import unittest
from src.communication.protocol import encode_states,parse_states

class ProtocolTest(unittest.TestCase):
    def test_roundtrip(self):
        states={1:(1.,2.,90.),2:(-1.,0.,-20.)}
        output=parse_states(encode_states(states))
        self.assertEqual(output,states)
    def test_bad_packet(self):
        with self.assertRaises(ValueError):parse_states(b'1 2 3')
    def test_empty(self):
        self.assertEqual(parse_states(b'\n'),{})
if __name__=='__main__':unittest.main()
