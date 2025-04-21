class _Node {
  val: number;
  next: _Node | null;
  random: _Node | null;

  constructor(val?: number, next?: _Node, random?: _Node) {
    this.val = val === undefined ? 0 : val;
    this.next = next === undefined ? null : next;
    this.random = random === undefined ? null : random;
  }
}

let map = new Map();

function copyRandomList(head: _Node | null): _Node | null {
  if (head == null) {
    return null;
  } else {
    let newNode = new _Node(head.val);
    map.set(head, newNode);
    let next = head.next;
    let random = head.random;
    if (map.get(newNode) != null) {
      newNode.next = map.get(newNode);
    } else {
      newNode.next = copyRandomList(next);
    }
    if (map.get(random) != null) {
      newNode.random = map.get(random);
    } else {
      newNode.random = copyRandomList(random);
    }
    return newNode;
  }
}

class ListNode {
  val: number;
  next: ListNode | null;
  constructor(val?: number, next?: ListNode | null) {
    this.val = val === undefined ? 0 : val;
    this.next = next === undefined ? null : next;
  }
}

function hasCycle(head: ListNode | null): boolean {
  let map = new Map();
  let res = false;
  while (head != null) {
    if (map.has(head)) {
      return true;
    }
    map.set(head, 1);
    head = head.next;
  }
  return res;
}
//142
function detectCycle(head: ListNode | null): ListNode | null {
  let map = new Map();
  let res = null;

  while (head != null) {
    if (map.has(head)) {
      return head;
    }
    map.set(head, 1);
    head = head.next;
  }
  return res;
}

class A { }
const a = new A();
const b = new A();
console.log(a == b);

function getIntersectionNode(
  headA: ListNode | null,
  headB: ListNode | null,
): ListNode | null {
  const map = new Map();
  while (headA != null) {
    map.set(headA, 0);
    headA = headA.next;
  }
  while (headB != null) {
    if (map.has(headB)) {
      return headB;
    }
    headB = headB.next;
  }
  return null;
}
//237
function deleteNode(node: ListNode | null): void {
  let next = node?.next;
  let p = node;
  while (next != null) {
    p!.val = next.val;
    if (next.next == null) {
      p!.next = null;
      break;
    }
    p = p!.next;
    next = next.next;
  }
}


/**
 * // This is the Iterator's API interface.
 * // You should not implement it, or speculate about its implementation
 * class Iterator {
 *      hasNext(): boolean {}
 *
 *      next(): number {}
 * }
 */


class Iterator {
  hasNext(): boolean {
    return true;
  }

  next(): number { 
    return 0;
  }
}
class PeekingIterator {
  public itr: Iterator<any>
  constructor(iterator: Iterator<any>) {
    this.itr
  }

  peek(): number {

  }

  next(): number {

  }

  hasNext(): boolean {

  }
}

/**
 * Your PeekingIterator object will be instantiated and called as such:
 * var obj = new PeekingIterator(iterator)
 * var param_1 = obj.peek()
 * var param_2 = obj.next()
 * var param_3 = obj.hasNext()
 */