from queue import Queue
from typing import List, Tuple

import manimlib as ml
import numpy as np


def make_job(job_id: str):
    job_square = ml.RoundedRectangle(width=2, height=1, corner_radius=0.2)
    job_square.set_fill(ml.BLUE, opacity=0.7)
    job_square.set_stroke(ml.BLUE_E, 1)

    job_text = ml.Text(job_id, font="Consolas", font_size=36)
    job_text.set_color(ml.WHITE)
    job_text.move_to(job_square.get_center())

    return ml.VGroup(job_square, job_text)


class JobQueue:
    def __init__(self, job_id: int, num_jobs: int, scene: ml.Scene):
        self.scene = scene
        self.job_id = job_id

        self.currently_displayed_jobs = ml.VGroup()
        self.ellipsis_queue = Queue()
        self.ellipsis = None

        for i in range(min(num_jobs, 5)):
            job = make_job(f"{job_id}_{i}")
            job.move_to(ml.UP * (min(num_jobs, 5) - i) * 1.1)
            self.currently_displayed_jobs.add(job)
        self.scene.add(self.currently_displayed_jobs)

        for i in range(5, num_jobs):
            self.ellipsis_queue.put(i)
        if num_jobs > 5:
            ellipsis = [ml.Circle(radius=0.1) for _ in range(3)]
            for e in ellipsis:
                e.set_fill(ml.BLUE, opacity=1)
                e.set_stroke(ml.BLUE, 1)
            ellipsis[0].move_to(ml.UP * 0.3)
            ellipsis[2].move_to(ml.DOWN * 0.3)
            ellipsis = ml.VGroup(*ellipsis)
            self.ellipsis = ellipsis
            self.scene.add(ellipsis)

        # For convenience in moving things around
        if self.ellipsis is not None:
            self.displayed_elements = ml.VGroup(
                self.currently_displayed_jobs, self.ellipsis
            )
        else:
            self.displayed_elements = self.currently_displayed_jobs

    def pop_job(self) -> Tuple[ml.VGroup, List[ml.Animation]]:
        """Pops a job from the queue and returns it
        If jobs exist in the hidden (ellipsis) queue, moves one
        into the currently displayed queue
        """
        anim_stack = []
        if len(self.currently_displayed_jobs) == 0:
            return None

        job = self.currently_displayed_jobs[0]
        # anim_stack.append(ml.FadeOut(job))
        self.currently_displayed_jobs.remove(job)

        # Shift jobs up
        for j in self.currently_displayed_jobs:
            j.generate_target()
            j.target.shift(ml.UP * 1.1)
            anim_stack.append(ml.MoveToTarget(j))

        # See if the ellipsis needs to go away

        if not self.ellipsis_queue.empty():
            new_job = make_job(f"{self.job_id}_{self.ellipsis_queue.get()}")
            new_job.move_to(self.currently_displayed_jobs[-1].get_center())
            self.currently_displayed_jobs.add(new_job)
            anim_stack.append(ml.FadeIn(new_job))
        else:
            if self.ellipsis is not None:
                anim_stack.append(ml.FadeOut(self.ellipsis))
                self.ellipsis = None

        return job, anim_stack

    def move_to(self, pos: np.ndarray):
        self.displayed_elements.move_to(pos)

    @property
    def num_jobs(self):
        return len(self.currently_displayed_jobs)

    @property
    def num_ellipsis(self):
        return self.ellipsis_queue.qsize()


class RunningJobs:
    def __init__(self, scene: ml.Scene):
        self.scene = scene
        self.displayed_jobs = ml.VGroup()
        self.scene.add(self.displayed_jobs)

        self.anchor_point = None

    def move_to(self, pos: np.ndarray):
        self.displayed_jobs.move_to(pos)

    def start_job(self, job: ml.VGroup) -> List[ml.Animation]:
        """Add a job to the running jobs list, pushing to the bottom"""

        if self.anchor_point is None:
            self.anchor_point = self.displayed_jobs.get_center()

        if len(self.displayed_jobs) == 0:
            target_pos = self.anchor_point
        else:
            target_pos = self.displayed_jobs[-1].get_center() + ml.DOWN * 1.1

        job.generate_target()
        job.target.move_to(target_pos)
        self.displayed_jobs.add(job)
        return [ml.MoveToTarget(job)]

    def finish_job(self) -> List[ml.Animation]:
        """Finishes a random job from the running jobs list"""
        n_active_jobs = len(self.displayed_jobs)
        if n_active_jobs == 0:
            return []

        animations = []
        rand_idx = int(np.random.randint(n_active_jobs))
        job_to_remove = self.displayed_jobs[rand_idx]

        # Shift lower jobs up
        for i in range(rand_idx + 1, n_active_jobs):
            job = self.displayed_jobs[i]
            job.generate_target()
            job.target.shift(ml.UP * 1.1)
            animations.append(ml.MoveToTarget(job))

        animations.append(ml.FadeOut(job_to_remove))
        self.displayed_jobs.remove(job_to_remove)
        return animations

    @property
    def num_jobs(self):
        return len(self.displayed_jobs)


class QueueAnimation(ml.Scene):

    def attempt_finish_job(self, running_jobs: RunningJobs, queued_jobs: JobQueue):
        if running_jobs.num_jobs == 0:
            return
        anims = running_jobs.finish_job()
        self.play(*anims)

    def attempt_start_job(self, running_jobs: RunningJobs, queued_jobs: JobQueue):
        if running_jobs.num_jobs >= 5 or queued_jobs.num_jobs == 0:
            return
        popped, anims = queued_jobs.pop_job()
        anims.extend(running_jobs.start_job(popped))
        self.play(*anims)

    def construct(self):
        queued_jobs = JobQueue(234112, 50, self)
        queued_jobs.move_to(ml.RIGHT * 3)

        running_jobs = RunningJobs(self)
        running_jobs.move_to(ml.LEFT * 3 + ml.UP * 2.5 * 1.1)

        self.gre_warning = ml.Text("QOSMaxGRESPerUser", font="Consolas", font_size=36)
        self.gre_warning.set_color(ml.WHITE)
        self.gre_warning.move_to(ml.LEFT * 3 + ml.DOWN * 3 * 1.1)

        running_jobs_text = ml.Text("Running Jobs", font="Consolas", font_size=36)
        running_jobs_text.set_color(ml.WHITE)
        running_jobs_text.move_to(ml.LEFT * 3 + ml.UP * 3.25 * 1.1)

        queued_jobs_text = ml.Text("Queued Jobs", font="Consolas", font_size=36)
        queued_jobs_text.set_color(ml.WHITE)
        queued_jobs_text.move_to(ml.RIGHT * 3 + ml.UP * 3.25 * 1.1)
        self.add(running_jobs_text, queued_jobs_text)

        self.wait(1)

        for _ in range(5):
            self.attempt_start_job(running_jobs, queued_jobs)

        # 50 job start and finish events
        events = ["s"] * 50 + ["f"] * 50
        job_start_times = np.cumsum(np.random.poisson(lam=0.3, size=50))
        job_finish_times = np.cumsum(np.random.poisson(lam=0.2, size=50))

        combined_times = np.concatenate([job_start_times, job_finish_times])
        sorting = combined_times.argsort()
        combined_times = combined_times[sorting]
        events = [events[i] for i in sorting]
        waits = np.insert(np.diff(combined_times), 0, 0)

        for w, event_type in zip(waits, events):
            self.wait(w)
            if event_type == "s":
                self.attempt_start_job(running_jobs, queued_jobs)
            else:
                self.attempt_finish_job(running_jobs, queued_jobs)

            self.check_max_gres_per_user(running_jobs)

    def check_max_gres_per_user(self, running_jobs: RunningJobs):
        if running_jobs.num_jobs >= 5:
            # self.play(ml.FadeIn(self.gre_warning))
            self.add(self.gre_warning)
        else:
            # self.play(ml.FadeOut(self.gre_warning))
            self.remove(self.gre_warning)
