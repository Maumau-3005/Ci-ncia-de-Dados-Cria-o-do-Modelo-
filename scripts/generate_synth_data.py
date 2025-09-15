#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    N = 1200

    def with_missing(x, p=0.05, codes=(9, 99, 777)):
        m = rng.random(N) < p
        x = x.astype(object)
        if m.any():
            x[m] = rng.choice(codes, size=m.sum())
        return x

    rfbing5 = rng.choice([1, 2], size=N, p=[0.7, 0.3])
    rfbing5 = with_missing(rfbing5, p=0.03)
    smoker3 = rng.choice([1, 2, 3, 4], size=N, p=[0.15, 0.10, 0.5, 0.25])
    smoker3 = with_missing(smoker3, p=0.03)

    drnk3ge5 = rng.integers(0, 20, size=N)
    drnk3ge5 = with_missing(drnk3ge5, p=0.02, codes=(9, 99))
    smokday2 = rng.choice([1, 2, 3], size=N, p=[0.1, 0.15, 0.75])
    smokday2 = with_missing(smokday2, p=0.02)

    _AGE80 = with_missing(rng.integers(18, 81, size=N))
    SEX = with_missing(rng.choice([1, 2], size=N))
    EDUCA = with_missing(rng.integers(1, 7, size=N))
    MARITAL = with_missing(rng.integers(1, 7, size=N))
    INCOME2 = with_missing(rng.integers(1, 9, size=N))
    GENHLTH = with_missing(rng.integers(1, 6, size=N))
    MENTHLTH = with_missing(rng.integers(0, 31, size=N))
    PHYSHLTH = with_missing(rng.integers(0, 31, size=N))
    _TOTINDA = with_missing(rng.choice([1, 2], size=N))
    EXEROFT1 = with_missing(rng.integers(0, 15, size=N))
    BPMEDS = with_missing(rng.choice([1, 2, 7, 9], size=N))
    BPHIGH4 = with_missing(rng.choice([1, 2, 3, 4], size=N))
    DIABETE3 = with_missing(rng.choice([1, 2, 3, 4], size=N))
    DIABAGE2 = with_missing(rng.integers(0, 80, size=N))
    CHOLCHK = with_missing(rng.choice([1, 2], size=N))
    HLTHPLN1 = with_missing(rng.choice([1, 2], size=N))
    PERSDOC2 = with_missing(rng.choice([1, 2, 3, 4], size=N))

    cols = {
        '_RFBING5': rfbing5,
        '_SMOKER3': smoker3,
        'DRNK3GE5': drnk3ge5,
        'SMOKDAY2': smokday2,
        '_AGE80': _AGE80,
        'SEX': SEX,
        'EDUCA': EDUCA,
        'MARITAL': MARITAL,
        'INCOME2': INCOME2,
        'GENHLTH': GENHLTH,
        'MENTHLTH': MENTHLTH,
        'PHYSHLTH': PHYSHLTH,
        '_TOTINDA': _TOTINDA,
        'EXEROFT1': EXEROFT1,
        'BPMEDS': BPMEDS,
        'BPHIGH4': BPHIGH4,
        'DIABETE3': DIABETE3,
        'DIABAGE2': DIABAGE2,
        'CHOLCHK': CHOLCHK,
        'HLTHPLN1': HLTHPLN1,
        'PERSDOC2': PERSDOC2,
    }
    df = pd.DataFrame(cols)

    os.makedirs('data', exist_ok=True)
    out = os.path.join('data', '2015.csv')
    df.to_csv(out, index=False)
    print('Wrote', out, 'shape=', df.shape)

if __name__ == '__main__':
    main()

