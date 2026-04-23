using UnityEngine;
using UnityEditor;

public class PoseCubeGenerator : MonoBehaviour
{
    private const float CubeEdgeLength = 0.08f;
    private const float ArrowGap = 0.004f;
    private const float ArrowShaftLength = 0.03f;
    private const float ArrowShaftThickness = 0.005f;
    private const float ArrowHeadLength = 0.014f;
    private const float ArrowHeadThickness = 0.005f;
    private const float ArrowHeadSpread = 0.006f;
    private const float ArrowHeadAngle = 35f;
    private const float LabelGap = 0.034f;
    private const float LabelCharacterSize = 0.02f;
    private const int LabelFontSize = 24;
    private const string ProjectRoot = "Assets/Project";
    private const string PrefabsFolder = ProjectRoot + "/Prefabs";
    private const string MeshesFolder = ProjectRoot + "/Meshes";
    private const string MaterialsFolder = ProjectRoot + "/Materials";
    private const string MeshPath = MeshesFolder + "/PoseCubeMesh.asset";

    [MenuItem("Tools/Create Pose Cube Prefab")]
    private static void CreatePoseCubePrefab()
    {
        GameObject poseCube = new GameObject("PoseCube");
        poseCube.transform.localScale = Vector3.one;

        GameObject cube = new GameObject("Cube");
        cube.name = "Cube";
        cube.transform.SetParent(poseCube.transform);
        cube.transform.localPosition = Vector3.zero;
        cube.transform.localRotation = Quaternion.identity;
        cube.transform.localScale = Vector3.one;

        Material redMaterial = CreateOrUpdateMaterial("PoseCube_Red", Color.red);
        Material greenMaterial = CreateOrUpdateMaterial("PoseCube_Green", Color.green);
        Material blueMaterial = CreateOrUpdateMaterial("PoseCube_Blue", Color.blue);

        MeshFilter meshFilter = cube.AddComponent<MeshFilter>();
        meshFilter.sharedMesh = CreateOrUpdatePoseCubeMesh();

        MeshRenderer renderer = cube.AddComponent<MeshRenderer>();
        Material[] materials = new Material[6];

        // Submesh order: Front, Back, Left, Right, Bottom, Top
        // Z axis faces -> Blue
        materials[0] = blueMaterial;
        materials[1] = blueMaterial;
        // X axis faces -> Red
        materials[2] = redMaterial;
        materials[3] = redMaterial;
        // Y axis faces -> Green
        materials[4] = greenMaterial;
        materials[5] = greenMaterial;

        renderer.sharedMaterials = materials;

        AddAxisArrow(
            poseCube.transform,
            CreateUniformMaterialArray(redMaterial),
            redMaterial.color,
            "X+",
            new Vector3(1f, 0f, 0f),
            new Vector3(0f, 1f, 0f),
            new Vector3(0f, 0f, -1f));

        AddAxisArrow(
            poseCube.transform,
            CreateUniformMaterialArray(greenMaterial),
            greenMaterial.color,
            "Y+",
            new Vector3(0f, 1f, 0f),
            new Vector3(1f, 0f, 0f),
            new Vector3(0f, 0f, 1f));

        AddAxisArrow(
            poseCube.transform,
            CreateUniformMaterialArray(blueMaterial),
            blueMaterial.color,
            "Z+",
            new Vector3(0f, 0f, 1f),
            new Vector3(1f, 0f, 0f),
            new Vector3(0f, -1f, 0f));

        EnsureFolder(ProjectRoot);
        EnsureFolder(PrefabsFolder);
        EnsureFolder(MeshesFolder);
        EnsureFolder(MaterialsFolder);

        string prefabPath = PrefabsFolder + "/PoseCube.prefab";
        if (AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath) != null)
            AssetDatabase.DeleteAsset(prefabPath);

        PrefabUtility.SaveAsPrefabAsset(poseCube, prefabPath);
        DestroyImmediate(poseCube);

        Debug.Log("PoseCube prefab created: " + prefabPath);
    }

    private static Mesh CreateOrUpdatePoseCubeMesh()
    {
        Mesh mesh = AssetDatabase.LoadAssetAtPath<Mesh>(MeshPath);
        if (mesh == null)
        {
            mesh = new Mesh();
            mesh.name = "PoseCubeMesh";
            AssetDatabase.CreateAsset(mesh, MeshPath);
        }

        float half = CubeEdgeLength * 0.5f;
        Vector3[] vertices =
        {
            new Vector3(-half, -half,  half),
            new Vector3( half, -half,  half),
            new Vector3( half,  half,  half),
            new Vector3(-half,  half,  half),

            new Vector3( half, -half, -half),
            new Vector3(-half, -half, -half),
            new Vector3(-half,  half, -half),
            new Vector3( half,  half, -half),

            new Vector3(-half, -half, -half),
            new Vector3(-half, -half,  half),
            new Vector3(-half,  half,  half),
            new Vector3(-half,  half, -half),

            new Vector3( half, -half,  half),
            new Vector3( half, -half, -half),
            new Vector3( half,  half, -half),
            new Vector3( half,  half,  half),

            new Vector3(-half, -half, -half),
            new Vector3( half, -half, -half),
            new Vector3( half, -half,  half),
            new Vector3(-half, -half,  half),

            new Vector3(-half,  half,  half),
            new Vector3( half,  half,  half),
            new Vector3( half,  half, -half),
            new Vector3(-half,  half, -half)
        };

        Vector3[] normals =
        {
            Vector3.forward, Vector3.forward, Vector3.forward, Vector3.forward,
            Vector3.back, Vector3.back, Vector3.back, Vector3.back,
            Vector3.left, Vector3.left, Vector3.left, Vector3.left,
            Vector3.right, Vector3.right, Vector3.right, Vector3.right,
            Vector3.down, Vector3.down, Vector3.down, Vector3.down,
            Vector3.up, Vector3.up, Vector3.up, Vector3.up
        };

        Vector2[] uv =
        {
            new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(0f, 1f),
            new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(0f, 1f),
            new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(0f, 1f),
            new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(0f, 1f),
            new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(0f, 1f),
            new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(0f, 1f)
        };

        mesh.Clear();
        mesh.vertices = vertices;
        mesh.normals = normals;
        mesh.uv = uv;
        mesh.subMeshCount = 6;
        mesh.SetTriangles(new[] { 0, 1, 2, 0, 2, 3 }, 0);
        mesh.SetTriangles(new[] { 4, 5, 6, 4, 6, 7 }, 1);
        mesh.SetTriangles(new[] { 8, 9, 10, 8, 10, 11 }, 2);
        mesh.SetTriangles(new[] { 12, 13, 14, 12, 14, 15 }, 3);
        mesh.SetTriangles(new[] { 16, 17, 18, 16, 18, 19 }, 4);
        mesh.SetTriangles(new[] { 20, 21, 22, 20, 22, 23 }, 5);
        mesh.RecalculateBounds();

        EditorUtility.SetDirty(mesh);
        return mesh;
    }

    private static void AddAxisArrow(
        Transform parent,
        Material[] materials,
        Color labelColor,
        string axisLabel,
        Vector3 axisDirection,
        Vector3 headSpreadDirection,
        Vector3 headRotationAxis)
    {
        float half = CubeEdgeLength * 0.5f;
        Vector3 shaftScale = Vector3.one * ArrowShaftThickness;
        Vector3 shaftPosition = axisDirection * (half + ArrowGap + (ArrowShaftLength * 0.5f));
        shaftScale = ApplyAxisLength(shaftScale, axisDirection, ArrowShaftLength);
        CreateArrowSegment(parent, axisLabel + "_Shaft", shaftPosition, Quaternion.identity, shaftScale, materials);

        Vector3 headScale = Vector3.one * ArrowHeadThickness;
        headScale = ApplyAxisLength(headScale, axisDirection, ArrowHeadLength);
        Vector3 headCenter = axisDirection * (half + ArrowGap + ArrowShaftLength + (ArrowHeadLength * 0.5f));
        Vector3 spreadOffset = headSpreadDirection * ArrowHeadSpread;

        CreateArrowSegment(
            parent,
            axisLabel + "_HeadA",
            headCenter + spreadOffset,
            Quaternion.AngleAxis(ArrowHeadAngle, headRotationAxis),
            headScale,
            materials);

        CreateArrowSegment(
            parent,
            axisLabel + "_HeadB",
            headCenter - spreadOffset,
            Quaternion.AngleAxis(-ArrowHeadAngle, headRotationAxis),
            headScale,
            materials);

        Vector3 labelPosition = axisDirection * (half + ArrowGap + ArrowShaftLength + ArrowHeadLength + LabelGap);
        CreateAxisLabel(parent, axisLabel + "_Label", axisLabel.ToLowerInvariant(), labelPosition, axisLabel, labelColor);
    }

    private static Vector3 ApplyAxisLength(Vector3 baseScale, Vector3 axisDirection, float length)
    {
        baseScale = ConvertSizeToMeshScale(baseScale);
        float scaledLength = length / CubeEdgeLength;

        if (Mathf.Abs(axisDirection.x) > 0.5f)
            baseScale.x = scaledLength;
        else if (Mathf.Abs(axisDirection.y) > 0.5f)
            baseScale.y = scaledLength;
        else
            baseScale.z = scaledLength;

        return baseScale;
    }

    private static Vector3 ConvertSizeToMeshScale(Vector3 size)
    {
        return new Vector3(
            size.x / CubeEdgeLength,
            size.y / CubeEdgeLength,
            size.z / CubeEdgeLength);
    }

    private static void CreateArrowSegment(
        Transform parent,
        string name,
        Vector3 localPosition,
        Quaternion localRotation,
        Vector3 localScale,
        Material[] materials)
    {
        GameObject segment = new GameObject(name);
        segment.transform.SetParent(parent);
        segment.transform.localPosition = localPosition;
        segment.transform.localRotation = localRotation;
        segment.transform.localScale = localScale;

        MeshFilter filter = segment.AddComponent<MeshFilter>();
        filter.sharedMesh = CreateOrUpdatePoseCubeMesh();

        MeshRenderer renderer = segment.AddComponent<MeshRenderer>();
        renderer.sharedMaterials = materials;
    }

    private static void CreateAxisLabel(
        Transform parent,
        string name,
        string text,
        Vector3 localPosition,
        string axisLabel,
        Color color)
    {
        GameObject label = new GameObject(name);
        label.transform.SetParent(parent);
        label.transform.localPosition = localPosition;
        label.transform.localRotation = GetFixedLabelRotation(axisLabel);
        label.transform.localScale = Vector3.one;

        TextMesh textMesh = label.AddComponent<TextMesh>();
        textMesh.text = text;
        textMesh.anchor = TextAnchor.MiddleCenter;
        textMesh.alignment = TextAlignment.Center;
        textMesh.fontSize = LabelFontSize;
        textMesh.characterSize = LabelCharacterSize;
        textMesh.color = color;
    }

    private static Quaternion GetFixedLabelRotation(string axisLabel)
    {
        switch (axisLabel)
        {
            case "x+":
                return Quaternion.LookRotation(Vector3.right, Vector3.up) * Quaternion.AngleAxis(180f, Vector3.up);
            case "y+":
                return Quaternion.LookRotation(Vector3.up, Vector3.left)
                    * Quaternion.AngleAxis(270f, Vector3.forward)
                    * Quaternion.AngleAxis(180f, Vector3.up);
            case "z+":
                return Quaternion.LookRotation(Vector3.forward, Vector3.up)
                    * Quaternion.AngleAxis(180f, Vector3.forward)
                    * Quaternion.AngleAxis(180f, Vector3.up);
            default:
                return Quaternion.identity;
        }
    }

    private static Material[] CreateUniformMaterialArray(Material material)
    {
        Material[] materials = new Material[6];
        for (int i = 0; i < materials.Length; i++)
            materials[i] = material;

        return materials;
    }

    private static Material CreateOrUpdateMaterial(string name, Color color)
    {
        EnsureFolder(ProjectRoot);
        EnsureFolder(MaterialsFolder);

        string materialPath = MaterialsFolder + "/" + name + ".mat";
        Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
        if (material == null)
        {
            material = new Material(Shader.Find("Standard"));
            AssetDatabase.CreateAsset(material, materialPath);
        }

        material.color = color;
        material.SetFloat("_Metallic", 0f);
        material.SetFloat("_Glossiness", 0f);
        EditorUtility.SetDirty(material);

        return material;
    }

    private static void EnsureFolder(string folderPath)
    {
        if (AssetDatabase.IsValidFolder(folderPath))
            return;

        string parentPath = System.IO.Path.GetDirectoryName(folderPath)?.Replace("\\", "/");
        if (!string.IsNullOrEmpty(parentPath))
            EnsureFolder(parentPath);

        string folderName = System.IO.Path.GetFileName(folderPath);
        AssetDatabase.CreateFolder(parentPath, folderName);
    }
}
