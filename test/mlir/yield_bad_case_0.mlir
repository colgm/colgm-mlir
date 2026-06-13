// verify error: yield must be inside colgm.if or colgm.for
func.func @bad_yield_outside() {
  colgm.yield
}
