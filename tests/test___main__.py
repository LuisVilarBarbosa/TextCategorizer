from subprocess import run
from tests.utils import config_file, decode
from text_categorizer import __main__, trainer, prediction_server

_expected_usage = 'Usage: python3 -m text_categorizer --trainer <configuration file>\n       python3 -m text_categorizer --prediction_server <configuration file> <port>\n'

def test_main(monkeypatch, capsys):
    all_args = [
        ['text_categorizer'],
        ['text_categorizer', 'invalid_arg'],
        ['text_categorizer', '--trainer', config_file, 'invalid_arg'],
        ['text_categorizer', '--prediction_server', config_file, '5000', 'invalid_arg'],
        ['text_categorizer', '--trainer', config_file],
        ['text_categorizer', '--prediction_server', config_file, '5000']
    ]
    with monkeypatch.context() as m:
        trainer_main_code = trainer.main.__code__
        prediction_server_main_code = prediction_server.main.__code__
        assert trainer_main_code.co_varnames[0:trainer_main_code.co_argcount] == ('config_filename',)
        assert prediction_server_main_code.co_varnames[0:prediction_server_main_code.co_argcount] == ('config_filename', 'port',)
        m.setattr("text_categorizer.trainer.main", lambda config_filename: None)
        m.setattr("text_categorizer.prediction_server.main", lambda config_filename, port: None)
        for i in range(len(all_args)):
            argv = all_args[i]
            __main__.main(argv)
            captured = capsys.readouterr()
            assert captured.out == (_expected_usage if i < len(all_args) - 2 else '')
            assert captured.err == ''

def test___main__():
    cp = run(args=['python', '-m', 'text_categorizer'], capture_output=True)
    assert cp.returncode == 0
    assert decode(cp.stdout) == _expected_usage
    assert decode(cp.stderr) == ''
