from patents_nlp.converte import to_single_pyfile
import os


def test_to_single_pyfile():
    outfilename = "./__allscript124.py"
    to_single_pyfile(outfile="./__allscript124.py")
    os.remove(outfilename)
