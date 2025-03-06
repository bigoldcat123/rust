# what’s different between threads and futures.


async blocks compile to anonymous futures, we can put each loop in an async block and have the runtime run them both to completion using the trpl::join function.

The synchronous Receiver::recv method in std::mpsc::channel blocks until it receives a message. The trpl::Receiver::recv method does not, because it is async. Instead of blocking, it hands control back to the runtime until either a message is received or the send side of the channel closes

This is a fundamental tradeoff: we can either deal with a dynamic number of futures with join_all, as long as they all have the same type, or we can deal with a set number of futures with the join functions or the join! macro, even if they have different types. 


That means if you do a bunch of work in an async block without an await point, that future will block any other futures from making progress.