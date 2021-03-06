import os


def to_single_pyfile(infolder: str = "./patents_nlp/",
                     outfile: str = "./allscript.py") -> None:
    """Converts all files in infolder into single python file that can be
    Easily submitted for kaggle competition.

    Args:
        infolder (str, optional): Folder where files be merged are stored.
            Defaults to "./patents_nlp/".
        outfile (str, optional): Target output merged filename.
            Defaults to "./allscript.py".
    """
    flist = os.listdir(infolder)
    flist = list(filter(lambda s: s != "main.py" and s !=
                        "converte.py" and not s.startswith("_") and
                        s != "cfg.py", flist))
    flist.insert(0, "cfg.py")
    flist.append("main.py")
    print("file order:", flist)
    with open(outfile, "w") as outf:
        for file in flist:
            with open(infolder + file, "r") as infile:
                while True:
                    line = infile.readline()
                    if not line:
                        break
                    split_import = line.split(' ')[:2]
                    if len(split_import) == 1:
                        outf.write(line)
                        continue
                    if (split_import[0] == "import" or
                       split_import[0] == "from"):
                        split_import[-1] = split_import[-1].split('\n')[0]
                        split_import[1] = split_import[1].split('.')[0]
                        if (split_import[1] == "patents_nlp"):
                            continue
                        else:
                            outf.write(line)
                    else:
                        if file == "main.py":
                            outf.write(line)
                            continue
                        splitline = line.split(' ')[:2]
                        if (splitline[0] == 'if' and
                           splitline[1][:8] == "__name__"):
                            break
                        outf.write(line)


if __name__ == "__main__":
    to_single_pyfile()
