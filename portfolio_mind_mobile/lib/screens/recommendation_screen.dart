import 'package:flutter/material.dart';
import '../models/recommendation.dart';
import '../services/api_service.dart';

class RecommendationScreen extends StatefulWidget {
  const RecommendationScreen({super.key});

  @override
  State<RecommendationScreen> createState() => _RecommendationScreenState();
}

class _RecommendationScreenState extends State<RecommendationScreen> {
  final ApiService _apiService = ApiService();
  late Future<Recommendation> _futureRecommendation;

  @override
  void initState() {
    super.initState();
    _futureRecommendation = _apiService.getRecommendation(1);
  }

  IconData _getAgentIcon(String name) {
    if (name.contains('Macro')) return Icons.public;
    if (name.contains('Behavior')) return Icons.psychology;
    if (name.contains('Allocation')) return Icons.auto_graph;
    if (name.contains('Risk')) return Icons.shield;
    if (name.contains('News')) return Icons.newspaper;
    return Icons.smart_toy;
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Recommendation>(
      future: _futureRecommendation,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }

        if (snapshot.hasError) {
          return Center(child: Text('Erreur: ${snapshot.error}'));
        }

        if (!snapshot.hasData) {
          return const Center(child: Text('Aucune donnée'));
        }

        final recommendation = snapshot.data!;

        return ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              'Recommandation IA',
              style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            ...recommendation.agents.map(
              (agent) => Card(
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                child: ListTile(
                  leading: Icon(_getAgentIcon(agent.name)),
                  title: Text(agent.name),
                  subtitle: Text(agent.description),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: Colors.deepPurple,
                borderRadius: BorderRadius.circular(18),
              ),
              child: Text(
                recommendation.summary,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}
