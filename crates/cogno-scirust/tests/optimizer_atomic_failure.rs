use cogno_scirust::{
    optim::{AdamW, AmsGrad},
    Optimizer,
};

macro_rules! atomic_cases {
    ($name:ident, $kind:ty) => {
        #[test]
        fn $name() {
            let initial = [1.0, 2.0];
            for gradient in [[0.1, f32::NAN], [0.1, f32::MAX]] {
                let mut opt = <$kind>::try_new(0.01, 2).unwrap();
                let before = format!("{opt:?}");
                let mut param = initial;
                assert!(opt.step(&mut param, &gradient).is_err());
                assert_eq!(param, initial);
                assert_eq!(format!("{opt:?}"), before);
                let mut clean = <$kind>::try_new(0.01, 2).unwrap();
                let mut expected = initial;
                clean.step(&mut expected, &[0.1, 0.2]).unwrap();
                opt.step(&mut param, &[0.1, 0.2]).unwrap();
                assert_eq!(param, expected);
                assert_eq!(format!("{opt:?}"), format!("{clean:?}"));
            }
            for field in 0..5 {
                let mut opt = <$kind>::try_new(0.01, 2).unwrap();
                match field {
                    0 => {
                        opt.state.m.pop();
                    }
                    1 => {
                        opt.state.v.pop();
                    }
                    2 => {
                        opt.state.v_hat.pop();
                    }
                    3 => opt.state.v[1] = -1.0,
                    _ => opt.state.step = u64::MAX,
                }
                let before = format!("{opt:?}");
                let mut param = initial;
                assert!(opt.step(&mut param, &[0.1, 0.2]).is_err());
                assert_eq!(param, initial);
                assert_eq!(format!("{opt:?}"), before);
            }
        }
    };
}
atomic_cases!(adamw_atomic_failure, AdamW);
atomic_cases!(amsgrad_atomic_failure, AmsGrad);
