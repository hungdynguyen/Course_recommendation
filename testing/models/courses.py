

complex_schema = {
    "type": "object",
    "properties": {
        "course_id": {"type": "string", "description": "Unique identifier like DATA101"},
        "title": {"type": "string", "description": "Official name of the course"},
        "overview": {"type": "string"},
        "difficulty_level": {"type": "string", "enum": ["Beginner", "Intermediate", "Advanced", "Expert"]},
        "duration": {
            "type": "object",
            "properties": {
                "total_hours": {"type": "number"},
                "credits": {"type": ["number", "null"]}
            },
            "required": ["total_hours"]
        },
        "knowledge_graph_alignment": {
            "type": "object",
            "properties": {
                "primary_domain": {"type": "string"},
                "sub_domains": {"type": "array", "items": {"type": "string"}},
                "key_concepts": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["primary_domain", "sub_domains", "key_concepts"]
        },
        "skill_outcomes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                    "category": {"type": "string", "enum": ["Hard Skill", "Soft Skill", "Tool/Software"]},
                    "target_proficiency_level": {"type": "integer"},
                    "bloom_taxonomy_level": {"type": "string", "enum": ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]},
                    "outcome_description": {"type": "string"}
                },
                "required": ["skill_name", "category", "target_proficiency_level", "bloom_taxonomy_level", "outcome_description"]
            }
        },
        "entry_requirements": {
             "type": "object",
             "properties": {
                 "prerequisite_courses": {"type": "array", "items": {"type": "string"}},
                 "minimum_entry_skills": {
                     "type": "array",
                     "items": {
                         "type": "object",
                         "properties": {
                             "skill_name": {"type": "string"},
                             "minimum_proficiency_level": {"type": "integer"}
                         },
                         "required": ["skill_name", "minimum_proficiency_level"]
                     }
                 }
             },
             "required": ["prerequisite_courses", "minimum_entry_skills"]
        },
    },
    "required": ["course_id", "title", "provider", "overview", "difficulty_level", "duration", "knowledge_graph_alignment", "skill_outcomes"]
}