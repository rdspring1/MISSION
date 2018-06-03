#ifndef CMS_ML_MP_QUEUE_H_
#define CMS_ML_MP_QUEUE_H_

#include <vector>
#include <mutex>
#include <thread>

/*
   MP_Queue - A queue that supports parallel reading and processing data
   A thread reads data to fill the queue until it is full
   A worker thread safely swaps the full queue with an empty queue
 */
template <typename T>
class mp_queue
{
		private:
				const size_t MAX = 1;
				const size_t FULL;
				std::mutex mtx;
				std::vector<T> q;

		public:
				mp_queue(size_t full_) : FULL(full_) {}

				void enqueue(T item)
				{
						while(q.size() >= MAX * FULL)
						{
								std::this_thread::sleep_for (std::chrono::seconds(1));
						}

						mtx.lock();
						q.emplace_back(item);
						mtx.unlock();
				}

				void retrieve(std::vector<T>& result)
				{
						assert(result.empty());
						mtx.lock();
						std::swap(q, result);
						mtx.unlock();
				}

				operator bool() const
				{
						return !q.empty();
				}

				size_t size() const
				{
						return FULL;
				}

				bool full() const
				{
						return q.size() >= FULL;
				}
};
#endif /* CMS_ML_MP_QUEUE_H_ */
