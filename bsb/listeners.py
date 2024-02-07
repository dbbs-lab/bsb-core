import abc
import sys
import traceback

from .services.pool import Job, JobStatus, PoolStatus


class Listener(abc.ABC):
    def __init__(self, fail_fast=True):
        self._ff = fail_fast

    @abc.abstractmethod
    def __call__(self, jobs: list[Job], status: PoolStatus):
        pass


class NonTTYTerminalListener(Listener):
    def __call__(self, jobs: list[Job], status: PoolStatus):
        if status == PoolStatus.STARTING:
            print(f"There are {len(jobs)} jobs in the queue.\n")
        elif status == PoolStatus.RUNNING:
            failed_jobs = [job for job in jobs if job.error]
            if failed_jobs:
                # No point in trying to raise more than 1 error
                if self._ff:
                    raise failed_jobs[0].error
            else:
                # todo: store state of jobs
                # todo: make a diff of job states
                # todo: print out any changed jobs
                job_pending = sum([1 for job in jobs if job.status == JobStatus.PENDING])
                job_queued = sum([1 for job in jobs if job.status == JobStatus.QUEUED])
        elif status == PoolStatus.ENDING:
            failed_jobs = [job for job in jobs if job.error]
            for job in failed_jobs:
                print(f"Job {job} failed:")
                traceback.print_exception(
                    type(job.error), job.error, job.error.__traceback__, file=sys.stderr
                )


class TTYTerminalListener(Listener):
    def __call__(self, jobs: list[Job], status: PoolStatus):
        if status == PoolStatus.STARTING:
            # todo: initialize fancy curses window
            pass
        elif status == PoolStatus.RUNNING:
            # todo: update window
            pass
        elif status == PoolStatus.ENDING:
            # todo: stop fancy curses window
            pass
