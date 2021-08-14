# import os
# os.chdir('..')

import pytest
from scr import train


@pytest.fixture
def test_init():
    tr = train.Train()
    return tr

def test_train(test_init):
    tr = test_init
    tr.train()
    tr.epochs = 1
    print(tr.train_df[tr.train_filtr])
    print(tr.model)