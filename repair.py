
def repair_bad_serie(tsh, engine, seriename, reparator, rewriter):
    logs = tsh.log(engine, names=(seriename,), diff=True)
    for idx, log in enumerate(logs):
        reparator(tsh, engine, log)
        rewriter(tsh, engine, idx, log)
