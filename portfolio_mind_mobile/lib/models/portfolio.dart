class AssetItem {
  final String name;
  final int percentage;

  AssetItem({
    required this.name,
    required this.percentage,
  });

  factory AssetItem.fromJson(Map<String, dynamic> json) {
    return AssetItem(
      name: json['name'],
      percentage: json['percentage'],
    );
  }
}

class Portfolio {
  final int userId;
  final List<AssetItem> assets;

  Portfolio({
    required this.userId,
    required this.assets,
  });

  factory Portfolio.fromJson(Map<String, dynamic> json) {
    return Portfolio(
      userId: json['user_id'],
      assets: (json['assets'] as List)
          .map((item) => AssetItem.fromJson(item))
          .toList(),
    );
  }
}
