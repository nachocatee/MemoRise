import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import { View, Text, Image, Pressable } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { styles } from "./GroupStyle";

type RootStackParamList = {
  FindGroup: undefined;
};

const MyGroupScreen = () => {
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();

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
        <View style={styles.titleContainer}>
          <Text style={styles.title}>내 그룹</Text>
        </View>
        <View style={styles.container}>
          <Image
            style={styles.group}
            source={require("../../../assets/icons/group.png")}
          />
          <Text style={styles.empty}>참여한 그룹이 없습니다</Text>
          <Pressable
            style={({ pressed }) =>
              pressed
                ? [styles.findGroupBtn, styles.pressed]
                : styles.findGroupBtn
            }
            onPress={() => navigation.navigate("FindGroup")}
          >
            <Text style={styles.btnText}>그룹 찾기</Text>
          </Pressable>
        </View>
      </View>
    </LinearGradient>
  );
};

export default MyGroupScreen;
