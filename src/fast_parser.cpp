#include "fast_parser.h"

fast_parser::fast_parser(const char* name) : status(true), newline(false), offset(0), idx(0), count(0), addr(nullptr), taddr(nullptr)
{
		pg_size = sysconf(_SC_PAGE_SIZE);
		fd = open(name, O_RDONLY);
		pg_size = sysconf(_SC_PAGE_SIZE);
}

fast_parser::~fast_parser()
{
		close(fd);
}

bool fast_parser::setup()
{
		if(idx >= pg_size)
		{
				destroy_mmap();
		}

		if(!addr && !taddr)
		{
				return initialize_mmap();
		}
		return true;
}

bool fast_parser::initialize_mmap()
{
		addr = mmap(NULL, pg_size, PROT_READ, MAP_FILE | MAP_PRIVATE | MAP_POPULATE, fd, offset);
		if (addr == MAP_FAILED)
		{
				std::cout << "MMAP Failure" << std::endl;
				return false;
		}

		if(madvise(addr, pg_size, MADV_SEQUENTIAL|MADV_WILLNEED) != 0) 
		{
				std::cerr << "Hint Failure" << std::endl;
				return false;
		}
		taddr = reinterpret_cast<char*>(addr);
		return true;
}

void fast_parser::destroy_mmap()
{
		idx = 0;
		munmap(addr, pg_size);
		offset += pg_size;
		addr = nullptr;
		taddr = nullptr;
}

data_t fast_parser::strtok(const char delimiter)
{
		data_t buf;
		memset(buf.data(), 0, buf.size());

		size_t cdx = 0;
		for(char v = this->operator++(0); status && v != delimiter && v != '\n'; v=this->operator++(0))
		{
				buf[cdx++] = v;
		}
		return buf;
}

std::vector<data_t> fast_parser::read(const char delimiter)
{
		std::vector<data_t> result;
		do
		{
				data_t item = this->strtok(delimiter);
				result.emplace_back(item);
		}
		while(status && !newline);
		return result;
}

size_t fast_parser::size() const
{
		return this->count;
}

char fast_parser::operator++(int)
{
		setup();
		char value = taddr[idx];
		status = (value != 0);
		newline = (value == '\n');
		count += (newline) ? 1 : 0;
		++idx;
		return value;
}

char fast_parser::operator*()
{
		setup();
		return taddr[idx];
}

fast_parser:: operator bool() const
{
		return status;
}
