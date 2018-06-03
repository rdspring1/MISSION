#ifndef CMS_ML_FAST_PARSER_H_
#define CMS_ML_FAST_PARSER_H_

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <vector>
#include <iostream>
#include <utility>
#include <stdio.h>
#include <string.h>
#include <assert.h>

typedef std::array<char, 32> data_t;

/*
   Fast Parser - Read data efficiently using Memory-Mapped I/O
 */
class fast_parser
{
		private:
				bool status;
				bool newline;

				size_t offset;
				size_t idx;
				size_t count;

				void* addr;
				char* taddr;

				size_t pg_size;
				int fd;

		public:
				fast_parser(const char*);
				~fast_parser();

				bool setup();
				bool initialize_mmap();
				void destroy_mmap();
				size_t size() const;
				data_t strtok(const char);
				std::vector<data_t> read(const char);

				char operator++(int);
				char operator*();
				operator bool() const;
};
#endif /* CMS_ML_FAST_PARSER_H_ */
