import 'package:flutter/material.dart';

class CustomInfoCard extends StatelessWidget {
  final String title;
  final String value;
  final IconData icon;

  const CustomInfoCard({
    super.key,
    required this.title,
    required this.value,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: Icon(icon),
        title: Text(title),
        subtitle: Text(value),
      ),
    );
  }
}
