import yaml
import random
import copy
from dataclasses import dataclass
from prettytable import PrettyTable

random.seed(12)


class TimeSlot:
    def __init__(self, day: str, time: int):
        self.day: str = day
        self.time: int = time

    def __eq__(self, value: object) -> bool:
        return value is not None and self.day == value.day and self.time == value.time

    def __str__(self) -> str:
        return f"{self.day}, {self.time}"

    def __repr__(self) -> str:
        return f"{self.day}, {self.time}"

    def __hash__(self) -> int:
        return hash((self.day, self.time))


class Subject:
    def __init__(self, name, hours):
        self.name: str = name
        self.hours: int = hours

    def __eq__(self, value: object) -> bool:
        return (
            value is not None and self.name == value.name and self.hours == value.hours
        )

    def __str__(self) -> str:
        return f"{self.name}"

    def __hash__(self) -> int:
        return hash((self.name, self.hours))


class Group:
    def __init__(self, name, capacity, subject_names):
        self.name: str = name
        self.capacity: int = capacity
        self.subject_names: list[str] = [subject_name for subject_name in subject_names]
        self.subjects: list[Subject] = None

    def __eq__(self, value: object) -> bool:
        return (
            value is not None
            and self.name == value.name
            and self.capacity == value.capacity
        )

    def __str__(self) -> str:
        return f"{self.name}"

    def __hash__(self) -> int:
        return hash((self.name, self.capacity))


class Lecturer:
    def __init__(self, name: str, can_teach_subjects_names: list[str]) -> None:
        self.name: str = name
        self.can_teach_subjects_names: list[str] = can_teach_subjects_names

    def __eq__(self, value: object) -> bool:
        return value is not None and self.name == value.name

    def __str__(self) -> str:
        return f"{self.name}"

    def __hash__(self) -> int:
        return hash(self.name)


class Hall:
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity

    def __eq__(self, value: object) -> bool:
        return (
            value is not None
            and self.name == value.name
            and self.capacity == value.capacity
        )

    def __str__(self) -> str:
        return f"{self.name}"

    def __hash__(self) -> int:
        return hash((self.name, self.capacity))


@dataclass(frozen=True)  # Immutable class
class Slot:
    group: Group
    subject: Subject
    lecturer: Lecturer
    hall: Hall
    time_slot: TimeSlot


class Parameters:
    def __init__(
        self,
        time_slots: list[TimeSlot],
        subjects: list[Subject],
        groups: list[Group],
        lecturers: list[Lecturer],
        halls: list[Hall],
    ):
        self.time_slots: list[TimeSlot] = time_slots
        self.subjects: list[Subject] = subjects
        self.groups: list[Group] = groups
        self.lecturers: list[Lecturer] = lecturers
        self.halls: list[Hall] = halls


class EvolutionParameters:
    def __init__(
        self,
        population_size: int,
        num_of_generations: int,
        mut_prob: float,
        elitism_ratio: float,
        tournament_size: int,
        fitness_func: callable,
        hall_prob: float = 0.2,
        lecturer_prob: float = 0.2,
        time_slot_prob: float = 0.2,
    ):
        self.population_size = population_size
        self.num_of_generations = num_of_generations
        self.mut_prob = mut_prob
        self.elitism_ratio = elitism_ratio
        self.tournament_size = tournament_size
        self.fitness_func = fitness_func
        self.hall_prob = hall_prob
        self.lecturer_prob = lecturer_prob
        self.time_slot_prob = time_slot_prob


class Schedule:
    def __init__(self, parameters: Parameters):
        self.grid: list[Slot] = []
        self.parameters = parameters

    def __str__(self) -> str:
        return "\n".join([str(slot) for slot in self.grid])

    def get_available_lecturers(self, time_slot: TimeSlot) -> list[Lecturer]:
        """
        Returns a list of available lecturers for the given time slot.
        """
        scheduled_slots = [slot for slot in self.grid if slot.time_slot == time_slot]
        scheduled_lecturers = {
            slot.lecturer for slot in scheduled_slots if slot.lecturer
        }
        available_lecturers = [
            lecturer
            for lecturer in self.parameters.lecturers
            if lecturer not in scheduled_lecturers
        ]
        return available_lecturers

    def get_available_halls(self, time_slot: TimeSlot) -> list[Hall]:
        """
        Returns a list of available halls for the given time slot.
        """
        scheduled_slots = [slot for slot in self.grid if slot.time_slot == time_slot]
        scheduled_halls = {slot.hall for slot in scheduled_slots if slot.hall}
        available_halls = [
            hall for hall in self.parameters.halls if hall not in scheduled_halls
        ]
        return available_halls

    def get_available_time_slots_with_hall_and_lecturer_and_group(
        self, hall: Hall, lecturer: Lecturer, group: Group
    ) -> list[TimeSlot]:
        """
        Returns a list of available time slots where the given hall and lecturer are both free.
        """
        scheduled_slots = [
            slot
            for slot in self.grid
            if slot.hall == hall or slot.lecturer == lecturer or slot.group == group
        ]

        occupied_time_slots = {slot.time_slot for slot in scheduled_slots}
        available_time_slots = [
            time_slot
            for time_slot in self.parameters.time_slots
            if time_slot not in occupied_time_slots
        ]
        return available_time_slots

    def mutate(self, evolution_params: EvolutionParameters):
        """
        Mutates the schedule with a given probability `mut_prob`.
        For each slot in the grid:
        - With probability `mut_prob`, independently apply one mutation (change hall, change lecturer, or change time slot)
        Each mutation has its own probability: `hall_prob`, `lecturer_prob`, `time_slot_prob`.
        """

        for i, slot in enumerate(self.grid):
            if random.random() > evolution_params.mut_prob:
                continue

            mutated_slot = slot

            if random.random() < evolution_params.hall_prob:
                available_halls = self.get_available_halls(slot.time_slot)
                if available_halls:
                    mutated_slot = Slot(
                        group=slot.group,
                        subject=slot.subject,
                        lecturer=slot.lecturer,
                        hall=random.choice(available_halls),
                        time_slot=slot.time_slot,
                    )

                    self.grid[i] = mutated_slot

            if random.random() < evolution_params.lecturer_prob:
                available_lecturers = self.get_available_lecturers(slot.time_slot)
                if available_lecturers:
                    mutated_slot = Slot(
                        group=slot.group,
                        subject=slot.subject,
                        lecturer=random.choice(available_lecturers),
                        hall=mutated_slot.hall,
                        time_slot=mutated_slot.time_slot,
                    )

                    self.grid[i] = mutated_slot

            if random.random() < evolution_params.time_slot_prob:
                available_time_slots = (
                    self.get_available_time_slots_with_hall_and_lecturer_and_group(
                        mutated_slot.hall, mutated_slot.lecturer, slot.group
                    )
                )
                if available_time_slots:
                    mutated_slot = Slot(
                        group=slot.group,
                        subject=slot.subject,
                        lecturer=mutated_slot.lecturer,
                        hall=mutated_slot.hall,
                        time_slot=random.choice(available_time_slots),
                    )

                    self.grid[i] = mutated_slot

    def count_total_windows(self) -> int:
        """
        Calculates the total number of "windows" (gaps) in the schedule across all groups.

        Returns:
            int: The total count of windows across all groups.
        """
        total_windows = 0

        for group in self.parameters.groups:
            group_windows = 0
            group_slots = [slot for slot in self.grid if slot.group == group]

            slots_by_day = {}
            for slot in group_slots:
                if slot.time_slot.day not in slots_by_day:
                    slots_by_day[slot.time_slot.day] = []
                slots_by_day[slot.time_slot.day].append(slot)

            for day, slots in slots_by_day.items():
                sorted_slots = sorted(slots, key=lambda s: s.time_slot.time)

                for i in range(len(sorted_slots) - 1):
                    current_lesson = sorted_slots[i].time_slot.time
                    next_lesson = sorted_slots[i + 1].time_slot.time

                    gap = next_lesson - current_lesson - 1
                    if gap > 0:
                        group_windows += gap

            total_windows += group_windows

        return total_windows

    def count_total_lecturer_windows(self) -> int:
        """
        Calculates the total number of "windows" (gaps) in the schedule across all lecturers.

        Returns:
            int: The total count of windows across all lecturers.
        """
        total_windows = 0

        for lecturer in self.parameters.lecturers:
            lecturer_windows = 0
            lecturer_slots = [slot for slot in self.grid if slot.lecturer == lecturer]

            slots_by_day = {}
            for slot in lecturer_slots:
                if slot.time_slot.day not in slots_by_day:
                    slots_by_day[slot.time_slot.day] = []
                slots_by_day[slot.time_slot.day].append(slot)

            for day, slots in slots_by_day.items():
                sorted_slots = sorted(slots, key=lambda s: s.time_slot.time)

                for i in range(len(sorted_slots) - 1):
                    current_lesson = sorted_slots[i].time_slot.time
                    next_lesson = sorted_slots[i + 1].time_slot.time

                    gap = next_lesson - current_lesson - 1
                    if gap > 0:
                        lecturer_windows += gap

            total_windows += lecturer_windows

        return total_windows

    def count_total_non_profile_slots(self) -> int:
        """
        Calculates the total number of slots across all lecturers where they are teaching a non-profile subject.

        Returns:
            int: The total count of non-profile slots across all lecturers.
        """
        total_non_profile_slots = 0

        for slot in self.grid:
            lecturer = slot.lecturer
            if lecturer and slot.subject.name not in lecturer.can_teach_subjects_names:
                total_non_profile_slots += 1

        return total_non_profile_slots

    def count_capacity_overflows(self) -> float:
        """
        Calculates the total capacity overflow penalty for halls where the group size exceeds hall capacity.

        Returns:
            float: The sum of overflow percentages for all cases where a hall's capacity is exceeded.
        """
        total_overflow_penalty = 0.0

        for slot in self.grid:
            group_size = slot.group.capacity
            hall_capacity = slot.hall.capacity

            if group_size > hall_capacity:
                # Calculate overflow percentage (e.g., if group size is 120 and hall capacity is 100, overflow is 20%)
                overflow_percentage = (group_size - hall_capacity) / hall_capacity
                total_overflow_penalty += overflow_percentage

        return total_overflow_penalty

    @classmethod
    def create_basic_schedule(cls, parameters: Parameters) -> "Schedule":
        """
        Creates a simple schedule that satisfies the hard constraints:
        - One lecturer can conduct one lesson at a time in one hall with one group.
        - One group can have one lesson at a time.
        - One hall can contain only one lesson at a time.

        This method does not aim for optimization, but creates a valid initial schedule.
        It returns a new Schedule object with the grid populated.

        Args:
            parameters (Parameters): The parameters required to generate the schedule.

        Returns:
            Schedule: A new Schedule object with the grid populated.
        """

        schedule = cls(parameters)
        schedule.grid = []

        for group in parameters.groups:
            shuffled_time_slots = iter(
                random.sample(parameters.time_slots, len(parameters.time_slots))
            )

            for subject_name in group.subject_names:
                subject = next(
                    (subj for subj in parameters.subjects if subj.name == subject_name),
                    None,
                )

                if subject is None:
                    continue

                for _ in range(subject.hours):
                    while True:
                        try:
                            time_slot = next(shuffled_time_slots)

                            # If this time slot is available, proceed with finding a hall and lecturer
                            available_halls = [
                                hall
                                for hall in parameters.halls
                                if hall
                                not in [
                                    slot.hall
                                    for slot in schedule.grid
                                    if slot.time_slot == time_slot
                                ]
                            ]

                            if not available_halls:
                                continue

                            hall = random.choice(available_halls)

                            # Get available lecturers at this time slot
                            available_lecturers = [
                                lecturer
                                for lecturer in parameters.lecturers
                                if lecturer
                                not in [
                                    slot.lecturer
                                    for slot in schedule.grid
                                    if slot.time_slot == time_slot
                                ]
                            ]

                            if not available_lecturers:
                                continue

                            lecturer = random.choice(available_lecturers)

                            slot = Slot(
                                group=group,
                                subject=subject,
                                lecturer=lecturer,
                                hall=hall,
                                time_slot=time_slot,
                            )
                            schedule.grid.append(slot)
                            break

                        except StopIteration:
                            print(
                                f"No available time slots for {group.name} and {subject.name}"
                            )
                            break

        return schedule


class EvolutionParameters:
    def __init__(
        self,
        population_size: int,
        num_of_generations: int,
        mut_prob: float,
        fitness_func: callable,
        selector_func: callable,
        hall_prob: float,
        lecturer_prob: float,
        time_slot_prob: float,
    ):
        self.population_size = population_size
        self.num_of_generations = num_of_generations
        self.mut_prob = mut_prob
        self.fitness_func = fitness_func
        self.selector_func = selector_func
        self.hall_prob = hall_prob
        self.lecturer_prob = lecturer_prob
        self.time_slot_prob = time_slot_prob


class GeneticSchedule:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    @classmethod
    def from_yaml(cls, file_path: str) -> "GeneticSchedule":
        try:
            with open(file_path, "r") as file:
                data = yaml.load(file, Loader=yaml.FullLoader)

                required_keys = [
                    "time_slots",
                    "subjects",
                    "groups",
                    "lecturers",
                    "halls",
                ]
                for key in required_keys:
                    if key not in data:
                        raise ValueError(f"Missing required key: {key} in YAML file.")

                time_slots = [TimeSlot(**time_slot) for time_slot in data["time_slots"]]
                subjects = [Subject(**subject) for subject in data["subjects"]]
                groups = [Group(**group) for group in data["groups"]]
                lecturers = [Lecturer(**lecturer) for lecturer in data["lecturers"]]
                halls = [Hall(**hall) for hall in data["halls"]]

                parameters = Parameters(time_slots, subjects, groups, lecturers, halls)

                return cls(parameters)

        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
        except ValueError as e:
            print(f"Error: {e}")

    def generate_population(self, size: int) -> list[Schedule]:
        """
        Generate a population of 'n' schedules using the create_basic_schedule method from the Schedule class.

        Args:
            n (int): The number of schedules to generate.

        Returns:
            list[Schedule]: A list of generated schedules.
        """
        population = []

        for _ in range(size):
            schedule = Schedule.create_basic_schedule(self.parameters)
            population.append(schedule)

        return population

    def evolve(self, evolution_params: EvolutionParameters) -> Schedule:
        """
        Evolves a population of schedules to optimize fitness.

        :param evolution_params: EvolutionParameters containing all parameters needed for evolution
        :return: Schedule with the best fitness after the evolution process
        """
        population = [
            Schedule.create_basic_schedule(self.parameters)
            for _ in range(evolution_params.population_size)
        ]

        table = PrettyTable()
        table.field_names = [
            "Generation",
            "Best Fitness",
            "Second Best Fitness",
            "Third Best Fitness",
        ]

        for generation in range(evolution_params.num_of_generations):
            for individual in population:
                individual.mutate(evolution_params)

            fitness_scores = [evolution_params.fitness_func(ind) for ind in population]
            top_3_fitness_before_selection = sorted(fitness_scores, reverse=True)[:3]

            table.add_row(
                [
                    generation + 1,
                    (
                        f"{top_3_fitness_before_selection[0]:.2f}"
                        if len(top_3_fitness_before_selection) > 0
                        else "N/A"
                    ),
                    (
                        f"{top_3_fitness_before_selection[1]:.2f}"
                        if len(top_3_fitness_before_selection) > 1
                        else "N/A"
                    ),
                    (
                        f"{top_3_fitness_before_selection[2]:.2f}"
                        if len(top_3_fitness_before_selection) > 2
                        else "N/A"
                    ),
                ]
            )

            print("\033c", end="")  # This clears the console
            print(table)

            population = evolution_parameters.selector_func(
                population,
                evolution_params.fitness_func,
            )

        best_schedule = max(population, key=evolution_params.fitness_func)
        return best_schedule


def save_schedule_to_yaml_student(schedule: Schedule, file_path: str) -> None:
    """
    Saves the given Schedule to a YAML file with lessons sorted by time slots for each group.

    :param schedule: The Schedule object to save
    :param file_path: Path to the output YAML file
    """
    schedule_data = {}

    for group in schedule.parameters.groups:
        group_lessons = [slot for slot in schedule.grid if slot.group == group]

        sorted_lessons = sorted(
            group_lessons, key=lambda slot: (slot.time_slot.day, slot.time_slot.time)
        )

        group_lessons_data = []
        for lesson in sorted_lessons:
            lesson_data = {
                "subject": lesson.subject.name,
                "lecturer": lesson.lecturer.name,
                "hall": lesson.hall.name,
                "time_slot": str(lesson.time_slot),
            }
            group_lessons_data.append(lesson_data)

        schedule_data[group.name] = group_lessons_data

    with open(file_path, "w") as file:
        yaml.dump(schedule_data, file, default_flow_style=False, allow_unicode=True)


import yaml


def save_schedule_to_yaml_lecturer(schedule: Schedule, file_path: str) -> None:
    """
    Saves the given Schedule to a YAML file with lessons sorted by time slots for each lecturer.

    :param schedule: The Schedule object to save
    :param file_path: Path to the output YAML file
    """
    schedule_data = {}

    for lecturer in schedule.parameters.lecturers:
        lecturer_lessons = [slot for slot in schedule.grid if slot.lecturer == lecturer]

        sorted_lessons = sorted(
            lecturer_lessons, key=lambda slot: (slot.time_slot.day, slot.time_slot.time)
        )

        lecturer_lessons_data = []
        for lesson in sorted_lessons:
            lesson_data = {
                "group": lesson.group.name,
                "subject": lesson.subject.name,
                "hall": lesson.hall.name,
                "time_slot": str(lesson.time_slot),
            }
            lecturer_lessons_data.append(lesson_data)

        schedule_data[lecturer.name] = lecturer_lessons_data

    with open(file_path, "w") as file:
        yaml.dump(schedule_data, file, default_flow_style=False, allow_unicode=True)


import yaml


def save_schedule_to_yaml_hall(schedule: Schedule, file_path: str) -> None:
    """
    Saves the given Schedule to a YAML file with lessons sorted by time slots for each hall.

    :param schedule: The Schedule object to save
    :param file_path: Path to the output YAML file
    """
    schedule_data = {}

    for hall in schedule.parameters.halls:
        hall_lessons = [slot for slot in schedule.grid if slot.hall == hall]

        sorted_lessons = sorted(
            hall_lessons, key=lambda slot: (slot.time_slot.day, slot.time_slot.time)
        )

        hall_lessons_data = []
        for lesson in sorted_lessons:
            lesson_data = {
                "group": lesson.group.name,
                "subject": lesson.subject.name,
                "lecturer": lesson.lecturer.name,
                "time_slot": str(lesson.time_slot),
            }
            hall_lessons_data.append(lesson_data)

        schedule_data[hall.name] = hall_lessons_data

    with open(file_path, "w") as file:
        yaml.dump(schedule_data, file, default_flow_style=False, allow_unicode=True)


def generate_selection_function(elitism_ratio: float = 0.1, tournament_size: int = 3):
    """
    Creates a selection function for a population with elitism and tournament selection.

    :param elitism_ratio: The ratio of elite individuals to retain from the population.
    :param tournament_size: The number of individuals to sample for tournament selection.
    :return: A function that returns the selected population.
    """

    def selector(population: list[Schedule], fitness_func: callable) -> list[Schedule]:
        population.sort(key=fitness_func, reverse=True)

        elitism_count = int(elitism_ratio * len(population))
        new_population = population[:elitism_count]

        while len(new_population) < len(population):
            tournament = random.sample(population, tournament_size)
            tournament.sort(key=fitness_func, reverse=True)
            new_population.append(copy.deepcopy(tournament[0]))

        return new_population

    return selector


def create_fittest_selector():
    """
    Creates a selection function that selects the fittest individual and replicates it
    to generate the entire new population.

    :return: A function that returns a new population containing only copies of the fittest individual.
    """

    def selector(population: list[Schedule], fitness_func: callable) -> list[Schedule]:
        population.sort(key=fitness_func, reverse=True)

        fittest_individual = population[0]

        new_population = [
            copy.deepcopy(fittest_individual) for _ in range(len(population))
        ]

        return new_population

    return selector


def generate_fitness_function(
    group_window_weight: float,
    lecturer_window_weight: float,
    non_profile_slot_weight: float,
    capacity_overflow_weight: float,
    distribution_penalty_weight: float = 0,
):
    def fitness(schedule: Schedule) -> float:
        total_group_windows = schedule.count_total_windows()
        total_lecturer_windows = schedule.count_total_lecturer_windows()
        total_non_profile_slots = schedule.count_total_non_profile_slots()
        total_capacity_overflow = schedule.count_capacity_overflows()

        # Distribution of lessons across time slots
        time_slot_counts = {
            time_slot: 0 for time_slot in schedule.parameters.time_slots
        }
        for slot in schedule.grid:
            time_slot_counts[slot.time_slot] += 1

        lesson_counts = list(time_slot_counts.values())
        max_count = max(lesson_counts)
        min_count = min(lesson_counts)

        # Distribution penalty: the larger the difference between the max and min, the worse the score
        distribution_penalty = max_count - min_count

        fitness_score = (
            group_window_weight * total_group_windows
            + lecturer_window_weight * total_lecturer_windows
            + non_profile_slot_weight * total_non_profile_slots
            + capacity_overflow_weight * total_capacity_overflow
            + distribution_penalty_weight * distribution_penalty
        ) / (
            group_window_weight
            + lecturer_window_weight
            + non_profile_slot_weight
            + capacity_overflow_weight
            + distribution_penalty_weight
        )

        return -1 * fitness_score  # Minimize penalties for optimization

    return fitness


fitness_func = generate_fitness_function(
    group_window_weight=10,
    lecturer_window_weight=7,
    non_profile_slot_weight=5,
    capacity_overflow_weight=20,
    distribution_penalty_weight=0,
)

# selector_func = generate_selection_function(elitism_ratio=0.1, tournament_size=15)
selector_func = create_fittest_selector()

evolution_parameters = EvolutionParameters(
    population_size=100,
    num_of_generations=50,
    mut_prob=0.1,
    fitness_func=fitness_func,
    selector_func=selector_func,
    hall_prob=0.2,
    lecturer_prob=0.2,
    time_slot_prob=0.2,
)


genetic_schedule = GeneticSchedule.from_yaml("schedule.yaml")
final_schedule = genetic_schedule.evolve(evolution_parameters)

# save_schedule_to_yaml_student(
#     Schedule.create_basic_schedule(genetic_schedule.parameters), "initial_student.yaml"
# )

save_schedule_to_yaml_student(final_schedule, "final_students_schedule.yaml")
save_schedule_to_yaml_lecturer(final_schedule, "final_lecturers_schedule.yaml")
save_schedule_to_yaml_hall(final_schedule, "final_halls_schedule.yaml")
