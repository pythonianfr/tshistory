
def test_info(engine, cli):
    r = cli('info', engine.url)
    out = r.output.strip().split('\n')

    assert len(out) == 3
    assert out[0].startswith('changeset count:')
    assert out[1].startswith('series count:')
    assert out[2].startswith('series names:')
