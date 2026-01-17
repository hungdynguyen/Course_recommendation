from typing import List, Dict, Optional
import logging
from collections import defaultdict, deque

from src.service_api.core.graph_db import Neo4jClient

logger = logging.getLogger(__name__)


class CourseRecommendationService:
    def __init__(self):
        self.neo4j = Neo4jClient()

    def close(self):
        self.neo4j.close()

    def recommend_courses(
        self, skill_ids: List[str], max_courses: int = 10
    ) -> Dict:
        """
        Recommend courses for target skills with prerequisite ordering.
        """
        # Get all courses teaching these skills
        courses = self._get_courses_for_skills(skill_ids)

        if not courses:
            return {
                "requested_skills": self._get_skill_details(skill_ids),
                "recommended_courses": [],
                "learning_path": [],
            }

        # Enrich with skill details
        enriched_courses = self._enrich_courses_with_skills(courses)

        # Order by prerequisites (topological sort)
        learning_path = self._create_learning_path(enriched_courses)

        # Limit results
        final_courses = enriched_courses[:max_courses]
        final_path = [c["course_id"] for c in final_courses]

        return {
            "requested_skills": self._get_skill_details(skill_ids),
            "recommended_courses": final_courses,
            "learning_path": final_path,
        }

    def get_courses_teaching_skill(
        self, skill_id: str, limit: int = 10
    ) -> List[Dict]:
        """
        Get all courses that teach a specific skill.
        """
        query = """
        MATCH (c:Course)-[t:TEACHES]->(s:Skill {skill_id: $skill_id})
        RETURN c.course_id AS course_id,
               c.course_title AS course_title,
               c.category AS category,
               t.similarity_score AS similarity_score
        ORDER BY t.similarity_score DESC
        LIMIT $limit
        """
        with self.neo4j.driver.session() as session:
            result = session.run(query, skill_id=skill_id, limit=limit)
            return [dict(record) for record in result]

    def get_course_details(self, course_id: str) -> Optional[Dict]:
        """
        Get detailed information about a course.
        """
        query = """
        MATCH (c:Course {course_id: $course_id})
        OPTIONAL MATCH (c)-[t:TEACHES]->(taught:Skill)
        OPTIONAL MATCH (c)-[r:REQUIRES]->(required:Skill)
        RETURN c.course_id AS course_id,
               c.course_title AS course_title,
               c.category AS category,
               collect(DISTINCT {skill_id: taught.skill_id, 
                                 label: taught.preferred_label,
                                 similarity: t.similarity_score}) AS taught_skills,
               collect(DISTINCT {skill_id: required.skill_id,
                                 label: required.preferred_label}) AS required_skills
        """
        with self.neo4j.driver.session() as session:
            result = session.run(query, course_id=course_id)
            record = result.single()
            if not record:
                return None
            
            data = dict(record)
            # Filter out None entries
            data["taught_skills"] = [s for s in data["taught_skills"] if s["skill_id"]]
            data["required_skills"] = [s for s in data["required_skills"] if s["skill_id"]]
            return data

    def _get_courses_for_skills(self, skill_ids: List[str]) -> List[Dict]:
        """
        Query Neo4j for courses teaching any of the target skills.
        """
        query = """
        MATCH (c:Course)-[t:TEACHES]->(s:Skill)
        WHERE s.skill_id IN $skill_ids
        RETURN DISTINCT c.course_id AS course_id,
               c.course_title AS course_title,
               c.category AS category
        """
        with self.neo4j.driver.session() as session:
            result = session.run(query, skill_ids=skill_ids)
            return [dict(record) for record in result]

    def _enrich_courses_with_skills(self, courses: List[Dict]) -> List[Dict]:
        """
        Add taught and required skills for each course.
        """
        course_ids = [c["course_id"] for c in courses]
        query = """
        UNWIND $course_ids AS cid
        MATCH (c:Course {course_id: cid})
        OPTIONAL MATCH (c)-[t:TEACHES]->(taught:Skill)
        OPTIONAL MATCH (c)-[r:REQUIRES]->(required:Skill)
        RETURN c.course_id AS course_id,
               collect(DISTINCT {skill_id: taught.skill_id,
                                 label: taught.preferred_label,
                                 similarity: t.similarity_score}) AS taught_skills,
               collect(DISTINCT {skill_id: required.skill_id,
                                 label: required.preferred_label}) AS required_skills
        """
        with self.neo4j.driver.session() as session:
            result = session.run(query, course_ids=course_ids)
            skills_map = {}
            for record in result:
                data = dict(record)
                # Filter None
                data["taught_skills"] = [s for s in data["taught_skills"] if s["skill_id"]]
                data["required_skills"] = [s for s in data["required_skills"] if s["skill_id"]]
                skills_map[record["course_id"]] = data

        # Merge back
        for course in courses:
            cid = course["course_id"]
            if cid in skills_map:
                course["taught_skills"] = skills_map[cid]["taught_skills"]
                course["required_skills"] = skills_map[cid]["required_skills"]
                # Calculate avg similarity
                taught = skills_map[cid]["taught_skills"]
                if taught:
                    scores = [s.get("similarity") for s in taught if s.get("similarity")]
                    course["similarity_score"] = sum(scores) / len(scores) if scores else None
                else:
                    course["similarity_score"] = None
            else:
                course["taught_skills"] = []
                course["required_skills"] = []
                course["similarity_score"] = None
            course["prerequisite_level"] = 0

        return courses

    def _create_learning_path(self, courses: List[Dict]) -> List[str]:
        """
        Order courses by prerequisite dependencies using topological sort.
        Courses with no prerequisites come first.
        """
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        course_map = {c["course_id"]: c for c in courses}

        for course in courses:
            cid = course["course_id"]
            in_degree[cid] = 0

        # If course A requires skills taught by course B, then B -> A
        for course in courses:
            required_skill_ids = {s["skill_id"] for s in course["required_skills"]}
            for other in courses:
                if other["course_id"] == course["course_id"]:
                    continue
                taught_skill_ids = {s["skill_id"] for s in other["taught_skills"]}
                if required_skill_ids & taught_skill_ids:
                    # other teaches something course requires
                    graph[other["course_id"]].append(course["course_id"])
                    in_degree[course["course_id"]] += 1

        # Topological sort (Kahn's algorithm)
        queue = deque([cid for cid in in_degree if in_degree[cid] == 0])
        sorted_order = []
        level = 0

        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                cid = queue.popleft()
                sorted_order.append(cid)
                course_map[cid]["prerequisite_level"] = level
                for neighbor in graph[cid]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            level += 1

        # Add any remaining (circular dependencies)
        remaining = [cid for cid in in_degree if cid not in sorted_order]
        sorted_order.extend(remaining)

        return sorted_order

    def _get_skill_details(self, skill_ids: List[str]) -> List[Dict]:
        """
        Get skill details from Neo4j.
        """
        query = """
        MATCH (s:Skill)
        WHERE s.skill_id IN $skill_ids
        RETURN s.skill_id AS skill_id,
               s.preferred_label AS label,
               s.description AS description
        """
        with self.neo4j.driver.session() as session:
            result = session.run(query, skill_ids=skill_ids)
            return [dict(record) for record in result]