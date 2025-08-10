@RestController
@RequestMapping("/api")
public class PredictionController {

    @PostMapping("/predict")
    public ResponseEntity<?> predictPerformance(@RequestBody EmployeeData data) {
        // Send data to Python Flask API
        RestTemplate restTemplate = new RestTemplate();
        String pythonApiUrl = "http://localhost:5000/predict";
        ResponseEntity<Map> response = restTemplate.postForEntity(pythonApiUrl, data, Map.class);
        return ResponseEntity.ok(response.getBody());
    }
}
