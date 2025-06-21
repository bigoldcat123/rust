// class _Node {
//   val: number;
//   next: _Node | null;
//   random: _Node | null;

//   constructor(val?: number, next?: _Node, random?: _Node) {
//     this.val = val === undefined ? 0 : val;
//     this.next = next === undefined ? null : next;
//     this.random = random === undefined ? null : random;
//   }
// }

// let map = new Map();

// function copyRandomList(head: _Node | null): _Node | null {
//   if (head == null) {
//     return null;
//   } else {
//     let newNode = new _Node(head.val);
//     map.set(head, newNode);
//     let next = head.next;
//     let random = head.random;
//     if (map.get(newNode) != null) {
//       newNode.next = map.get(newNode);
//     } else {
//       newNode.next = copyRandomList(next);
//     }
//     if (map.get(random) != null) {
//       newNode.random = map.get(random);
//     } else {
//       newNode.random = copyRandomList(random);
//     }
//     return newNode;
//   }
// }

// class ListNode {
//   val: number;
//   next: ListNode | null;
//   constructor(val?: number, next?: ListNode | null) {
//     this.val = val === undefined ? 0 : val;
//     this.next = next === undefined ? null : next;
//   }
// }

// function hasCycle(head: ListNode | null): boolean {
//   let map = new Map();
//   let res = false;
//   while (head != null) {
//     if (map.has(head)) {
//       return true;
//     }
//     map.set(head, 1);
//     head = head.next;
//   }
//   return res;
// }
// //142
// function detectCycle(head: ListNode | null): ListNode | null {
//   let map = new Map();
//   let res = null;

//   while (head != null) {
//     if (map.has(head)) {
//       return head;
//     }
//     map.set(head, 1);
//     head = head.next;
//   }
//   return res;
// }

// class A { }
// const a = new A();
// const b = new A();
// console.log(a == b);

// function getIntersectionNode(
//   headA: ListNode | null,
//   headB: ListNode | null,
// ): ListNode | null {
//   const map = new Map();
//   while (headA != null) {
//     map.set(headA, 0);
//     headA = headA.next;
//   }
//   while (headB != null) {
//     if (map.has(headB)) {
//       return headB;
//     }
//     headB = headB.next;
//   }
//   return null;
// }
// //237
// function deleteNode(node: ListNode | null): void {
//   let next = node?.next;
//   let p = node;
//   while (next != null) {
//     p!.val = next.val;
//     if (next.next == null) {
//       p!.next = null;
//       break;
//     }
//     p = p!.next;
//     next = next.next;
//   }
// }

//427
// class _Node {
//   val: boolean
//   isLeaf: boolean
//   topLeft: _Node | null
//   topRight: _Node | null
//   bottomLeft: _Node | null
//   bottomRight: _Node | null
//   constructor(val?: boolean, isLeaf?: boolean, topLeft?: _Node, topRight?: _Node, bottomLeft?: _Node, bottomRight?: _Node) {
//     this.val = (val === undefined ? false : val)
//     this.isLeaf = (isLeaf === undefined ? false : isLeaf)
//     this.topLeft = (topLeft === undefined ? null : topLeft)
//     this.topRight = (topRight === undefined ? null : topRight)
//     this.bottomLeft = (bottomLeft === undefined ? null : bottomLeft)
//     this.bottomRight = (bottomRight === undefined ? null : bottomRight)
//   }
// }
// function construct(grid: number[][]): _Node | null {
//   if (grid.length == 1) {
//     return null
//   } else {
//     return consturct_dfs(0, grid.length, 0, grid.length, grid)
//   }
// };
// function consturct_dfs(r_start: number, r_end: number, c_start: number, c_end: number, grid: number[][]): _Node {

//   if (r_end - r_start == 1) {
//     return new _Node(grid[r_start][c_start] == 1, true);
//   }

//   const r_mid = (r_start + r_end) / 2;
//   const c_mid = (c_start + c_end) / 2;

//   const top_left_r_start = r_start;
//   const top_left_r_end = r_mid;

//   const top_right_r_start = r_start;
//   const top_right_r_end = r_mid;

//   const bottom_left_r_start = r_mid;
//   const bottom_left_r_end = r_end;

//   const bottom_right_r_start = r_mid;
//   const bottom_right_r_end = r_end;

//   const top_left_c_start = c_start;
//   const top_left_c_end = c_mid;

//   const top_right_c_start = c_mid;
//   const top_right_c_end = c_end;

//   const bottom_left_c_start = c_start;
//   const bottom_left_c_end = c_mid;

//   const bottom_right_c_start = c_mid;
//   const bottom_right_c_end = c_end;

//   let top_left = consturct_dfs(top_left_r_start, top_left_r_end, top_left_c_start, top_left_c_end, grid);
//   let top_right = consturct_dfs(top_right_r_start, top_right_r_end, top_right_c_start, top_right_c_end, grid);
//   let bottom_left = consturct_dfs(bottom_left_r_start, bottom_left_r_end, bottom_left_c_start, bottom_left_c_end, grid);
//   let bottom_right = consturct_dfs(bottom_right_r_start, bottom_right_r_end, bottom_right_c_start, bottom_right_c_end, grid);
//   if (top_left.isLeaf && top_right.isLeaf && bottom_left.isLeaf && bottom_right.isLeaf
//     && top_left.val == top_right.val && bottom_left.val == bottom_right.val && top_left == bottom_left
//   ) {
//     return new _Node(top_left.val, true, top_left, top_right, bottom_left, bottom_right)
//   } else {
//     return new _Node(top_left.val, false, top_left, top_right, bottom_left, bottom_right)
//   }
// }

//429
// class _Node {
//   val: number
//   children: _Node[]

//   constructor(v: number) {
//     this.val = v;
//     this.children = [];
//   }
// }



// function levelOrder(root: _Node): number[][] {
//   const q = new Array<_Node>()
//   q.push(root);

//   const res = new Array();

//   while (q.length != 0) {
//     const p = new Array<number>();
//     let p_queue = q.splice(0, q.length)

//     while (true) {
//       const next = p_queue.shift()
//       if (!next) {
//         break;
//       }
//       q.push(...next.children)
//       p.push(next.val);
//     }
//     res.push(p)
//   }

//   return res
// };

// //430
// class _Node {
//   val: number
//   prev: _Node | null
//   next: _Node | null
//   child: _Node | null

//   constructor(val?: number, prev?: _Node, next?: _Node, child?: _Node) {
//     this.val = (val === undefined ? 0 : val);
//     this.prev = (prev === undefined ? null : prev);
//     this.next = (next === undefined ? null : next);
//     this.child = (child === undefined ? null : child);
//   }
// }



// function flatten(head: _Node | null): _Node | null {
//   if (head == null) {
//     return null
//   }
//   let { start, end } = flatten_dfs(head!)
//   return start;
// };

// function flatten_dfs(start: _Node): {
//   start: _Node,
//   end: _Node
// } {
//   const s = start;

//   while (start.next != null || start.child != null) {
//     if (start.child != null) {
//       const res = flatten_dfs(start.child!);
//       start.child = null;
//       if (start.next == null) {

//         res.end.next = start.next;

//         start.next = res.start;
//         res.start.prev = start;
//         start = res.end;
//         break;
//       }
//       res.end.next = start.next;
//       start.next.prev = res.end;

//       start.next = res.start
//       res.start.prev = start;

//       start = res.end
//     }
//     start = start.next!
//   }
//   return {
//     start: s,
//     end: start
//   }
// }

// class PeekingIterator {
//     iter:Iterator
//     n:number | null
//     constructor(iterator: Iterator) {
//         this.iter = iterator
//         if (iterator.hasNext()) {
//             this.n = iterator.next();
//         }else {
//             this.n = null
//         }
//     }

//     peek(): number {
//         return this.n!
//     }

//     next(): number {
//         const n = this.n!;
//         if (this.iterator.hasNext()) {
//             this.n = this.iterator.next();
//         } else {
//             this.n = null
//         }
//         return n!
//     }

//     hasNext(): boolean {
//         return this.n != null
//     }
// }




// class _Node {
//     val: boolean
//     isLeaf: boolean
//     topLeft: _Node | null
//     topRight: _Node | null
//     bottomLeft: _Node | null
//     bottomRight: _Node | null
//     constructor(val?: boolean, isLeaf?: boolean, topLeft?: _Node, topRight?: _Node, bottomLeft?: _Node, bottomRight?: _Node) {
//         this.val = (val === undefined ? false : val)
//         this.isLeaf = (isLeaf === undefined ? false : isLeaf)
//         this.topLeft = (topLeft === undefined ? null : topLeft)
//         this.topRight = (topRight === undefined ? null : topRight)
//         this.bottomLeft = (bottomLeft === undefined ? null : bottomLeft)
//         this.bottomRight = (bottomRight === undefined ? null : bottomRight)
//     }
// }



// function dfs(quadTree1: _Node, quadTree2: _Node) {
//     if (quadTree1.isLeaf) {
//         if (quadTree1.val != true) {
//             quadTree1.bottomLeft = quadTree2.bottomLeft
//             quadTree1.bottomRight = quadTree2.bottomRight
//             quadTree1.topLeft = quadTree2.topLeft
//             quadTree1.topRight = quadTree2.topRight
//         }
//     } else if (quadTree2.isLeaf) {
//         if (quadTree2.val == true) {
//             quadTree1.val = true;
//             quadTree1.isLeaf = true
//             quadTree1.bottomLeft = null
//             quadTree1.bottomRight = null
//             quadTree1.topLeft = null
//             quadTree1.topRight = null
//         }
//     } else {
//         quadTree1.val ||= quadTree2.val;
//         dfs(quadTree1.topLeft!, quadTree2.topLeft!);
//         dfs(quadTree1.topRight!, quadTree2.topRight!);
//         dfs(quadTree1.bottomLeft!, quadTree2.bottomLeft!);
//         dfs(quadTree1.bottomRight!, quadTree2.bottomRight!);
//         if (quadTree1.topLeft!.isLeaf && quadTree1.topRight!.isLeaf && quadTree1.bottomLeft!.isLeaf && quadTree1.bottomRight!.isLeaf &&
//             quadTree1.topLeft!.val == quadTree1.topRight!.val && quadTree1.topRight!.val == quadTree1.bottomLeft!.val && quadTree1.bottomLeft!.val == quadTree1.bottomRight!.val) {
//             quadTree1.val = quadTree1.bottomLeft!.val
//             quadTree1.isLeaf = true;
//             quadTree1.bottomLeft = null
//             quadTree1.bottomRight =null
//             quadTree1.topLeft = null
//             quadTree1.topRight = null
//         }
//     }
// }

// function intersect(quadTree1: _Node | null, quadTree2: _Node | null): _Node | null {
//     dfs(quadTree1!, quadTree2!);
//     return quadTree1
// };



class _Node {
    val: number
    children: _Node[]

    constructor(val?: number, children?: _Node[]) {
        this.val = (val === undefined ? 0 : val)
        this.children = (children === undefined ? [] : children)
    }
}



function maxDepth(root: _Node | null): number {
    if (root == null) {
        return 0
    }else {
        let res = 1;
        for  (const e of root.children) {
            res = Math.max(res,maxDepth(e) + 1);
        }
        return res 
    }
};