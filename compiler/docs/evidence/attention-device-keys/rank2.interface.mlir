module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %A = merlin_iface.tensor {name = "A", role = "input"} : tensor<3x8xi8>
  %B = merlin_iface.tensor {name = "B", role = "input"} : tensor<5x8xi8>
  %Y = merlin_iface.attention_qk %A, %B {name = "Y", output_dtype = "i32", epilogue = []} : (tensor<3x8xi8>, tensor<5x8xi8>) -> tensor<3x5xi32>
}
