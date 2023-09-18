import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import { View, Text, Image, Pressable } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import { calculateDynamicWidth } from "../../constants/dynamicSize";
import Colors from "../../constants/colors";
import CancelHeader from "../../components/Header/CancelHeader";
import ProfilePic from "../../components/ProfilePic";
import { styles } from "./MenuStyle";

type RootStackParamList = {
  SavedMemo: undefined;
  AllMemo: undefined;
  MyMemo: undefined;
  MyGroup: undefined;
  FindGroup: undefined;
  ModifyInfo: undefined;
};

const MenuScreen = () => {
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();

  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <CancelHeader />
      <View style={styles.container}>
        <View style={{ marginBottom: calculateDynamicWidth(4) }}>
          <ProfilePic />
        </View>
        {/* 유저 닉네임 */}
        <Text style={styles.text}>권소정</Text>
        {/* 유저 이메일 */}
        <View style={styles.emailBox}>
          <Image
            source={require("../../assets/image/kakao_sm.png")}
            style={styles.kakao}
          />
          <Text style={styles.email}>flfk33@naver.com</Text>
        </View>
        {/* 노트 이미지 */}
        <View style={{ marginTop: calculateDynamicWidth(25) }}>
          <View>
            <Image
              source={require("../../assets/image/note.png")}
              style={styles.note}
            />
          </View>
          {/* 메뉴 contents */}
          <View style={styles.menuList}>
            <Pressable
              style={[
                styles.menuItem,
                { marginBottom: calculateDynamicWidth(7) },
              ]}
              onPress={() => navigation.navigate("SavedMemo")}
            >
              {({ pressed }) => (
                <>
                  <Image
                    source={require("../../assets/icons/bookmark.png")}
                    style={styles.bookmark}
                  />
                  <View>
                    <Text style={styles.text}>저장된 메모</Text>
                    {pressed && <View style={styles.highlight}></View>}
                  </View>
                </>
              )}
            </Pressable>
            <Pressable
              style={styles.menuItem}
              onPress={() => navigation.navigate("AllMemo")}
            >
              {({ pressed }) => (
                <>
                  <View
                    style={[
                      styles.dotIcon,
                      styles.menuIcon,
                      { backgroundColor: Colors.blue500 },
                    ]}
                  ></View>
                  <View>
                    <Text style={styles.text}>전체 메모</Text>
                    {pressed && <View style={styles.highlight}></View>}
                  </View>
                </>
              )}
            </Pressable>
            <Pressable
              style={styles.menuItem}
              onPress={() => navigation.navigate("MyMemo")}
            >
              {({ pressed }) => (
                <>
                  <View
                    style={[
                      styles.dotIcon,
                      styles.menuIcon,
                      { backgroundColor: "#9CBDFF" },
                    ]}
                  ></View>
                  <View>
                    <Text style={styles.text}>내 메모</Text>
                    {pressed && <View style={styles.highlight}></View>}
                  </View>
                </>
              )}
            </Pressable>

            <View style={styles.line}></View>

            <Pressable
              style={styles.menuItem}
              onPress={() => navigation.navigate("MyGroup")}
            >
              {({ pressed }) => (
                <>
                  <View
                    style={[
                      styles.dotIcon,
                      styles.menuIcon,
                      { backgroundColor: "#DAA418" },
                    ]}
                  ></View>
                  <View>
                    <Text style={styles.text}>내 그룹</Text>
                    {pressed && <View style={styles.highlight}></View>}
                  </View>
                </>
              )}
            </Pressable>
            <Pressable
              style={styles.menuItem}
              onPress={() => navigation.navigate("FindGroup")}
            >
              {({ pressed }) => (
                <>
                  <View
                    style={[
                      styles.dotIcon,
                      styles.menuIcon,
                      { backgroundColor: Colors.primary300 },
                    ]}
                  ></View>
                  <View>
                    <Text style={styles.text}>그룹 찾기</Text>
                    {pressed && <View style={styles.highlight}></View>}
                  </View>
                </>
              )}
            </Pressable>

            <View style={styles.line}></View>

            <Pressable
              style={[styles.menuItem, { marginTop: calculateDynamicWidth(7) }]}
              onPress={() => navigation.navigate("ModifyInfo")}
            >
              {({ pressed }) => (
                <>
                  <Image
                    source={require("../../assets/icons/setting.png")}
                    style={styles.bookmark}
                  />
                  <View>
                    <Text style={styles.text}>회원정보 수정</Text>
                    {pressed && <View style={styles.highlight}></View>}
                  </View>
                </>
              )}
            </Pressable>
          </View>
        </View>
      </View>
    </LinearGradient>
  );
};

export default MenuScreen;
