

import java.util.TreeMap
import java.sql.Array
import kotlin.math.max
import kotlin.math.min
class DisjointSet(val size:Int) {
    var cc = size
    val fa = MutableList(size) { it }
    fun find(x:Int):Int {
        if (fa[x] != x) {
            fa[x] = find(fa[x])
        }
        return fa[x]
    }
    fun union(from:Int,to:Int):Boolean {
        val a = find(from)
        val b = find(to)
        if (a == b) {
            return false
        }
        fa[a] = b
        cc -= 1
        return true
    }
}


class Solution {
    fun getFactors(m:Int):List<MutableList<Int>> {
        val fac = List(m + 10) { _ -> mutableListOf<Int>() }
        for (i in 2..m) {
            if (fac[i].isEmpty()) {
                var j = i
                while (j <= m) {
                    fac[j].add(i)
                    j += i
                }
            }
        }
        return fac
    }
    fun canTraverseAllPairs(nums: IntArray): Boolean {
        val max = nums.max()
        val d_set = DisjointSet(nums.size + max + 1)
        val fac = getFactors(max)
        for (i in 0..<nums.size) {
            for (f in fac[nums[i]]) {
                d_set.union(nums[i], nums.size + f)
            }
        }
        val h_set = HashSet<Int>()
        for (i in 0..<nums.size) {
            h_set.add(d_set.find(i))
        }
        return h_set.size == 1
    }
    fun maxFrequency(nums: IntArray, k: Int, numOperations: Int): Int {
        val cnt = HashMap<Int,Int>()
        val diff = TreeMap<Int,Int>()
        for (n in nums) {
            cnt.put(n, cnt.get(n)?:0 + 1)
            diff.putIfAbsent(n, 0);
            diff.put(n - k, diff.get(n - k)?:0 + 1)
            diff.put(n + k + 1, diff.get(n + k + 1)?:0 - 1)
        }
        var ans = 0;
        var sumD = 0;
        for ((key,v) in diff) {
            println(key)
            println(v)
            sumD += v;
            ans = max(ans,min(sumD,cnt.get(key)?:0 + numOperations))
        }
        return ans
    }
}
