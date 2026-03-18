import 'package:flutter/material.dart';
import '../utils/app_constants.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("About"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [

            const Text(
              "Portfolio Mind",
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 20),

            const ListTile(
              leading: Icon(Icons.person),
              title: Text("Author"),
              subtitle: Text("Eya Khalfallah"),
            ),

            const ListTile(
              leading: Icon(Icons.school),
              title: Text("Project"),
              subtitle: Text("Master PFE"),
            ),

            const ListTile(
              leading: Icon(Icons.cloud),
              title: Text("Backend"),
              subtitle: Text("AWS EC2 + FastAPI + Docker"),
            ),

            const ListTile(
              leading: Icon(Icons.code),
              title: Text("Technologies"),
              subtitle: Text("Flutter / DevSecOps / AI Agents"),
            ),

            const ListTile(
              leading: Icon(Icons.public),
              title: Text("Elastic IP"),
              subtitle: Text("98.88.54.142"),
            ),

            const ListTile(
              leading: Icon(Icons.info),
              title: Text("Version"),
              subtitle: Text("1.0.0"),
            ),
          ],
        ),
      ),
    );
  }
}
