use std::{
    num::NonZero,
    sync::{Arc, Mutex, atomic},
    task::{Context, Poll, Waker},
};

use num_rational::Ratio;

struct InnerLock<T> {
    result: Option<T>,
    waiters: Vec<Waker>,
    progress_callbacks: Vec<Box<dyn FnMut(Ratio<u32>) + Send>>,

    #[cfg(not(target_has_atomic = "32"))]
    current: u32,
}

struct Inner<T> {
    total: NonZero<u32>,

    #[cfg(target_has_atomic = "32")]
    current: atomic::AtomicU32,

    #[cfg(target_has_atomic = "32")]
    flags: atomic::AtomicU32,

    lock: Mutex<InnerLock<T>>,
}

#[cfg(target_has_atomic = "32")]
const FLAG_READY: u32 = 1 << 0;

#[cfg(target_has_atomic = "32")]
const FLAG_HAS_PROGRESS_CALLBACKS: u32 = 1 << 1;

pub enum Status<T> {
    Pending(Ratio<u32>),
    Ready(T),
}

pub struct Deferred<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Deferred<T> {
    pub fn progress(&self) -> Ratio<u32> {
        #[cfg(target_has_atomic = "32")]
        let current = self.inner.current.load(atomic::Ordering::Acquire);

        #[cfg(not(target_has_atomic = "32"))]
        let current = self.inner.lock.lock().unwrap().current;

        Ratio::new(current.min(self.inner.total.get()), self.inner.total.get())
    }

    pub fn status(&self) -> Status<T> {
        #[cfg(target_has_atomic = "32")]
        {
            let flags = self.inner.flags.load(atomic::Ordering::Acquire);
            if flags & FLAG_READY != 0 {
                let mut guard = self.inner.lock.lock().unwrap();
                match guard.result.take() {
                    Some(result) => Status::Ready(result),
                    None => {
                        let current = self.inner.current.load(atomic::Ordering::Acquire);
                        Status::Pending(Ratio::new(current, self.inner.total.get()))
                    }
                }
            } else {
                let current = self.inner.current.load(atomic::Ordering::Acquire);
                Status::Pending(Ratio::new(current, self.inner.total.get()))
            }
        }

        #[cfg(not(target_has_atomic = "32"))]
        {
            let mut guard = self.inner.lock.lock().unwrap();
            match guard.result.take() {
                Some(result) => Status::Ready(result),
                None => Status::Pending(Ratio::new(guard.current, self.inner.total.get())),
            }
        }
    }

    /// Polls the deferred for completion.
    /// Returns `Poll::Ready(result)` if the deferred is complete,
    /// or `Poll::Pending` if it is still in progress.
    pub fn poll(&self, cx: &mut Context<'_>) -> Poll<T> {
        #[cfg(target_has_atomic = "32")]
        {
            let flags = self.inner.flags.load(atomic::Ordering::Acquire);
            if flags & FLAG_READY != 0 {
                let mut guard = self.inner.lock.lock().unwrap();
                match guard.result.take() {
                    Some(result) => Poll::Ready(result),
                    None => Poll::Pending, // This case should not happen, but we return Pending to avoid panicking
                }
            } else {
                let mut guard = self.inner.lock.lock().unwrap();
                guard.waiters.push(cx.waker().clone());
                Poll::Pending
            }
        }

        #[cfg(not(target_has_atomic = "32"))]
        {
            let mut guard = self.inner.lock.lock().unwrap();
            let current = guard.current;
            match guard.result.take() {
                Some(result) => Poll::Ready(result),
                None => {
                    guard.waiters.push(cx.waker().clone());
                    Poll::Pending
                }
            }
        }
    }

    pub fn on_progress<F>(&self, callback: F)
    where
        F: FnMut(Ratio<u32>) + Send + 'static,
    {
        {
            let mut guard = self.inner.lock.lock().unwrap();
            guard.progress_callbacks.push(Box::new(callback));
        }

        #[cfg(target_has_atomic = "32")]
        self.inner
            .flags
            .fetch_or(FLAG_HAS_PROGRESS_CALLBACKS, atomic::Ordering::Release);
    }
}

pub struct Promise<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Promise<T> {
    pub fn update_progress(&mut self, current: u32) {
        debug_assert!(
            current <= self.inner.total.get(),
            "Progress cannot exceed total"
        );

        #[cfg(target_has_atomic = "32")]
        {
            self.inner.current.store(current, atomic::Ordering::Release);
            let flags = self.inner.flags.load(atomic::Ordering::Acquire);

            if flags & FLAG_HAS_PROGRESS_CALLBACKS == 0 {
                return;
            }

            let mut guard = self.inner.lock.lock().unwrap();
            for callback in guard.progress_callbacks.iter_mut() {
                callback(Ratio::new(current, self.inner.total.get()));
            }
        }

        #[cfg(not(target_has_atomic = "32"))]
        {
            let mut guard = self.inner.lock.lock().unwrap();
            guard.current = current;

            for callback in guard.progress_callbacks.iter_mut() {
                callback(Ratio::new(current, self.inner.total.get()));
            }
        }
    }

    pub fn complte(self, result: T) {
        #[cfg(target_has_atomic = "32")]
        {
            self.inner
                .current
                .store(self.inner.total.get(), atomic::Ordering::Release);
        }

        let mut guard = self.inner.lock.lock().unwrap();

        #[cfg(not(target_has_atomic = "32"))]
        {
            guard.current = self.inner.total.get();
        }

        // Set the result
        guard.result = Some(result);

        // Invoke progress callbacks with 100% progress
        for callback in guard.progress_callbacks.iter_mut() {
            callback(Ratio::new(self.inner.total.get(), self.inner.total.get()));
        }

        // Wake all waiters
        for waiter in guard.waiters.drain(..) {
            waiter.wake();
        }
    }
}

/// Creates a new deferred and its associated promise.
///
/// The `total` parameter specifies the total progress required for the deferred to be considered complete.
pub fn deferred_promise<T>(total: NonZero<u32>) -> (Deferred<T>, Promise<T>) {
    let inner = Arc::new(Inner {
        total,
        lock: Mutex::new(InnerLock {
            result: None,
            waiters: Vec::new(),
            progress_callbacks: Vec::new(),

            #[cfg(not(target_has_atomic = "32"))]
            current: 0,
        }),

        #[cfg(target_has_atomic = "32")]
        current: atomic::AtomicU32::new(0),

        #[cfg(target_has_atomic = "32")]
        flags: atomic::AtomicU32::new(0),
    });

    (
        Deferred {
            inner: inner.clone(),
        },
        Promise { inner },
    )
}

pub fn defer_spawn<F, T>(f: F) -> Deferred<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let (deferred, promise) = deferred_promise::<T>(const { NonZero::new(1).unwrap() });

    std::thread::spawn(move || {
        let result = f();
        promise.complte(result);
    });

    deferred
}

pub fn defer_spawn_fold<I, F, T>(iter: I, init: T, fold: F) -> Deferred<T>
where
    I: IntoIterator<Item = T>,
    I::IntoIter: Send + 'static,
    F: FnMut(T, I::Item) -> T + Send + 'static,
    T: Send + 'static,
{
    let iter = iter.into_iter();
    let (lower, upper) = iter.size_hint();
    let mut total_expected = upper.unwrap_or(lower);

    let (deferred, promise) = deferred_promise::<T>(const { NonZero::new(u32::MAX).unwrap() });

    std::thread::spawn(move || {
        let mut promise = promise;
        let mut fold = fold;
        let mut acc = init;
        let mut iter = iter;
        let mut processed = 0usize;

        while let Some(item) = iter.next() {
            acc = fold(acc, item);
            if processed == usize::MAX {
                promise.update_progress(u32::MAX - 1);
                while let Some(item) = iter.next() {
                    acc = fold(acc, item);
                }
                break;
            }

            processed += 1;

            let (lower, upper) = iter.size_hint();
            let more_expected = upper.unwrap_or(lower);

            total_expected = total_expected.max(processed + more_expected);

            let progress = 1.0 - more_expected as f64 / total_expected.max(1) as f64;

            let progress_u32 = (progress.clamp(0.0, 1.0) * u32::MAX as f64) as u32;
            promise.update_progress(progress_u32);
        }

        promise.complte(acc);
    });

    deferred
}
