#[test]
fn t1() {
    let a = vec![4,3,1,2,7,8,6];
    let mut a = a.into_iter().enumerate().collect::<Vec<_>>();
    a.sort_by(|a,b|a.1.cmp(&b.1));
    println!("{:?}",a);
    let mut visited = vec![false;a.len()];
    for i in 0..a.len() {
        if visited[i] {
            continue;
        }
        let mut i = i;
        while !visited[i] {
            println!("{} -> {}",i,a[i].0);
            visited[i] = true;
            i = a[i].0;
        }
    }
}
