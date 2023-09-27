// 라이브러리
import React from "react";
import { RouteProp } from "@react-navigation/native";
import { View, Text, Image } from "react-native";
import LinearGradient from "react-native-linear-gradient";

// 타입
import { RootStackParamList } from "../../../App";

// 컴포넌트
import GoBackHeader from "../../../components/Header/GoBackHeader";

// 스타일
import { styles } from "./MemoStyle";

type MenuMemoScreenProps = {
  route: RouteProp<RootStackParamList, "MenuMemo">;
};

const MemuMemo: React.FC<MenuMemoScreenProps> = ({ route }) => {
  const { menuStatus } = route.params;

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      {menuStatus === "saved" && (
        <View>
          <Text style={styles.title}>저장된 메모</Text>
          <View style={styles.container}>
            <Image
              style={styles.icon}
              source={require("../../../assets/icons/memo_empty.png")}
            />
            <Text style={styles.empty}>저장된 메모가 없습니다</Text>
          </View>
        </View>
      )}
      {menuStatus === "all" && (
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
      )}
      {menuStatus === "my" && (
        <View>
          <Text style={styles.title}>내 메모</Text>
          <View style={styles.container}>
            <Image
              style={styles.icon}
              source={require("../../../assets/icons/memo_empty.png")}
            />
            <Text style={styles.empty}>작성한 메모가 없습니다</Text>
          </View>
        </View>
      )}
    </LinearGradient>
  );
};

export default MemuMemo;
