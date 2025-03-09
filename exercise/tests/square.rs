use exercise::square::{ difference, square_of_sum, sum_of_squares};

#[test]
fn square_of_sum_1() {
    assert_eq!(1, square_of_sum(1));
}
#[test]
// #[ignore]
fn square_of_sum_5() {
    assert_eq!(225, square_of_sum(5));
}
#[test]
// #[ignore]
fn square_of_sum_100() {
    assert_eq!(25_502_500, square_of_sum(100));
}
#[test]
// #[ignore]
fn sum_of_squares_1() {
    assert_eq!(1, sum_of_squares(1));
}
#[test]
// #[ignore]
fn sum_of_squares_5() {
    assert_eq!(55, sum_of_squares(5));
}
#[test]
// #[ignore]
fn sum_of_squares_100() {
    assert_eq!(338_350, sum_of_squares(100));
}
#[test]
// #[ignore]
fn difference_1() {
    assert_eq!(0, difference(1));
}
#[test]
// #[ignore]
fn difference_5() {
    assert_eq!(170, difference(5));
}
#[test]
// #[ignore]
fn difference_100() {
    assert_eq!(25_164_150, difference(100));
}


