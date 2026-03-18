import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';
import '../utils/app_constants.dart';

class RecommendationScreen extends StatefulWidget {
  const RecommendationScreen({super.key});

  @override
  State<RecommendationScreen> createState() => _RecommendationScreenState();
}

class _RecommendationScreenState extends State<RecommendationScreen> {
  final ApiService apiService = ApiService();

  Map<String, dynamic>? recommendation;
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    loadRecommendation();
  }

  Future<void> loadRecommendation() async {
    try {
      final data = await apiService.getRecommendation(1);

      setState(() {
        recommendation = data;
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("AI Recommendation"),
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : recommendation == null
              ? const Center(child: Text("No recommendation"))
              : buildRecommendation(),
    );
  }

  Widget buildRecommendation() {
    final agents = recommendation!["agents"];

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [

          Card(
            elevation: 4,
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [

                  const Text(
                    "AI Portfolio Recommendation",
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  const SizedBox(height: 10),

                  Text(recommendation!["summary"]),

                  const SizedBox(height: 20),

                  const Text(
                    "Risk Score",
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),

                  const SizedBox(height: 5),

                  const LinearProgressIndicator(
                    value: 0.65,
                    minHeight: 10,
                  ),

                  const SizedBox(height: 20),

                  const Text(
                    "Confidence Score",
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),

                  const SizedBox(height: 5),

                  const LinearProgressIndicator(
                    value: 0.82,
                    minHeight: 10,
                    color: Colors.green,
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 30),

          const Text(
            "AI Agents",
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 10),

          ...agents.map<Widget>((agent) {
            return Card(
              child: ListTile(
                leading: const Icon(Icons.smart_toy),
                title: Text(agent["name"]),
                subtitle: Text(agent["description"]),
              ),
            );
          }).toList(),

          const SizedBox(height: 30),

          const Text(
            "Recommended Actions",
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 10),

          const Card(
            child: ListTile(
              leading: Icon(Icons.trending_up),
              title: Text("Increase ETF exposure"),
            ),
          ),

          const Card(
            child: ListTile(
              leading: Icon(Icons.currency_bitcoin),
              title: Text("Reduce crypto allocation"),
            ),
          ),

          const Card(
            child: ListTile(
              leading: Icon(Icons.workspace_premium),
              title: Text("Add gold hedge"),
            ),
          ),
        ],
      ),
    );
  }
}
