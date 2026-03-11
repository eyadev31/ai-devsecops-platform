import 'package:flutter/material.dart';

class RecommendationScreen extends StatelessWidget {
  const RecommendationScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: ListView(
        children: const [
          Text(
            'Recommandation IA',
            style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 16),

          Card(
            child: ListTile(
              leading: Icon(Icons.public),
              title: Text('Agent 1 - Macro Context Analyzer'),
              subtitle: Text('Analyse du contexte macroéconomique et marché'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.psychology),
              title: Text('Agent 2 - User Behavior Profiler'),
              subtitle: Text('Analyse du profil investisseur'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.auto_graph),
              title: Text('Agent 3 - Portfolio Allocation Optimizer'),
              subtitle: Text('Optimisation de l’allocation du portefeuille'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.shield),
              title: Text('Agent 4 - Risk Oversight & Validation'),
              subtitle: Text('Contrôle du risque et validation'),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(Icons.newspaper),
              title: Text('Agent 5 - News Sentiment Agent'),
              subtitle: Text('Analyse des actualités et du sentiment'),
            ),
          ),

          SizedBox(height: 20),

          Card(
            color: Colors.deepPurpleAccent,
            child: Padding(
              padding: EdgeInsets.all(16),
              child: Text(
                'Recommandation actuelle : portefeuille diversifié, profil modéré, exposition équilibrée entre actions, ETF, crypto et cash.',
                style: TextStyle(color: Colors.white, fontSize: 16),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
