import { View, Text, Image } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { styles } from "./MemoStyle";

const AllMemoScreen = () => {
  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View>
        <Text style={styles.title}>전체 메모</Text>
        <View style={styles.container}>
          <Image
            style={styles.icon}
            source={require("../../../assets/icons/memo_empty.png")}
          />
          <Text style={styles.empty}>메모가 없습니다</Text>
        </View>
      </View>
    </LinearGradient>
  );
};

export default AllMemoScreen;
