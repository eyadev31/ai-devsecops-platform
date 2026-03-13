class AgentInfo {
  final String name;
  final String description;

  AgentInfo({
    required this.name,
    required this.description,
  });

  factory AgentInfo.fromJson(Map<String, dynamic> json) {
    return AgentInfo(
      name: json['name'],
      description: json['description'],
    );
  }
}

class Recommendation {
  final String summary;
  final List<AgentInfo> agents;

  Recommendation({
    required this.summary,
    required this.agents,
  });

  factory Recommendation.fromJson(Map<String, dynamic> json) {
    return Recommendation(
      summary: json['summary'],
      agents: (json['agents'] as List)
          .map((item) => AgentInfo.fromJson(item))
          .toList(),
    );
  }
}
