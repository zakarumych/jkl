use jkl::max_rects::{Heuristic, MaximalRectangles};

fn main() {
    let mut pack = MaximalRectangles::new(32, 16);
    pack.with_quantization(2, 2)
        .with_allow_rotation(true)
        .with_heuristic(Heuristic::BottomLeft);

    let items = vec![
        (10, 5),
        (7, 3),
        (8, 4),
        (5, 2),
        (9, 6),
        (3, 3),
        (4, 2),
        (6, 5),
        (10, 3),
        (7, 2),
        (2, 6),
        (1, 4),
        (6, 8),
        (5, 4),
        (8, 4),
        (4, 8),
        (9, 2),
        (2, 5),
    ];

    for (w, h) in items {
        match pack.insert(w, h) {
            Some(p) => {
                println!("Placed: {:?}", p);
                println!();
            }
            None => println!("Failed to place {}x{}", w, h),
        }
    }

    println!("Occupancy: {:.2}%", pack.occupancy() * 100.0);
    println!();

    println!("Used:");
    for r in pack.used() {
        println!("{:?}", r);
    }
    println!("Free:");
    for r in pack.free() {
        println!("{:?}", r);
    }

    for y in 0..pack.height() {
        for x in 0..pack.width() {
            let mut symbol = '.';

            for (idx, r) in pack.used().enumerate() {
                if r.contains_point(x, y) {
                    assert_eq!(symbol, '.');
                    symbol = char::from_u32('A' as u32 + idx as u32).unwrap();
                }
            }

            print!("{}", symbol);
        }

        println!();
    }
}
