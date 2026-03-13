class UserProfile {
  final int id;
  final String name;
  final String riskLevel;
  final String horizon;
  final String objective;

  UserProfile({
    required this.id,
    required this.name,
    required this.riskLevel,
    required this.horizon,
    required this.objective,
  });

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      id: json['id'],
      name: json['name'],
      riskLevel: json['risk_level'],
      horizon: json['horizon'],
      objective: json['objective'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'risk_level': riskLevel,
      'horizon': horizon,
      'objective': objective,
    };
  }
}
