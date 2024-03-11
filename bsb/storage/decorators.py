def on_main(prep=None, ret=None):
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            r = None
            self.comm.barrier()
            if self.comm.get_rank() == 0:
                r = f(self, *args, **kwargs)
            elif prep:
                prep(self, *args, **kwargs)
            self.comm.barrier()
            if not ret:
                return self.comm.bcast(r, root=0)
            else:
                return ret(self, *args, **kwargs)

        return wrapper

    return decorator


def on_main_until(until, prep=None, ret=None):
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            global _procpass
            r = None
            self.comm.barrier()
            if self.comm.get_rank() == 0:
                r = f(self, *args, **kwargs)
            elif prep:
                prep(self, *args, **kwargs)
            self.comm.barrier()
            while not until(self, *args, **kwargs):
                pass
            if not ret:
                return self.comm.bcast(r, root=0)
            else:
                return ret(self, *args, **kwargs)

        return wrapper

    return decorator


__all__ = ["on_main", "on_main_until"]
