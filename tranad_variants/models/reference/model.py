from ...bootstrap import activate_baseline
activate_baseline()
from TranADPlus.src.models import TranAD


class ReferenceTranAD(TranAD):
    output_count = 2
    variant = "reference"
