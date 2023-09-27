package com.tjjhtjh.memorise.domain.user.controller;

import com.tjjhtjh.memorise.domain.memo.service.MemoService;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MyMemoResponse;
import com.tjjhtjh.memorise.domain.team.service.TeamService;
import com.tjjhtjh.memorise.domain.team.service.dto.response.MyTeamListResponse;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.UserService;
import com.tjjhtjh.memorise.domain.user.service.dto.request.JoinRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.request.UpdateUserInfoRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.response.JoinResponse;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UpdateUserInfoResponse;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserInfoResponse;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserListResponse;
import com.tjjhtjh.memorise.global.file.service.AwsS3Service;
import com.tjjhtjh.memorise.global.file.service.dto.CreateFileRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/user")
public class UserController {

    private final UserService userService;
    private final TeamService teamService;
    private final MemoService memoService;
    private final AwsS3Service awsS3Service;

    private static String dirName = "profile-image";

    @PostMapping("/upload")
    public ResponseEntity<CreateFileRequest> uploadMultipleFile(@RequestPart MultipartFile file) {
        return ResponseEntity.ok(awsS3Service.uploadMultiFile(file, dirName));
    }

    @PostMapping
    public ResponseEntity<JoinResponse> join(@RequestBody JoinRequest joinRequest) {
        User user = userService.join(joinRequest);
        return ResponseEntity.ok(new JoinResponse(true, user.getUserSeq()));
    }

    @PutMapping("/{userSeq}")
    public ResponseEntity<UpdateUserInfoResponse> update(@PathVariable Long userSeq, @RequestBody UpdateUserInfoRequest updateUserInfoRequest) {
        userService.updateUserInfo(userSeq, updateUserInfoRequest);
        return ResponseEntity.ok(new UpdateUserInfoResponse(true));
    }

    @GetMapping("/{userSeq}")
    public ResponseEntity<UserInfoResponse> getUserInfo(@PathVariable Long userSeq) {
        return ResponseEntity.ok(new UserInfoResponse(userService.getUserInfo(userSeq)));
    }

    @GetMapping("/list")
    public ResponseEntity<List<UserListResponse>> getUserList(@RequestParam(required = false) String keyword) {
        return ResponseEntity.ok(userService.getUserList(keyword));
    }

    @GetMapping("/{userSeq}/my-teams")
    public ResponseEntity<List<MyTeamListResponse>> getMyTeamList(@PathVariable Long userSeq) {
        return ResponseEntity.ok(teamService.getMyTeamList(userSeq));
    }

    @GetMapping("/{userSeq}/my-memos")
    public ResponseEntity<List<MyMemoResponse>> myMemoList(@PathVariable Long userSeq){
        return new ResponseEntity<>(memoService.myMemoList(userSeq) ,HttpStatus.OK);
    }

    @GetMapping("/{userSeq}/memos")
    public ResponseEntity<List<MyMemoResponse>> allMyMemoList(@PathVariable Long userSeq){
        return new ResponseEntity<>(memoService.allMyMemoList(userSeq) ,HttpStatus.OK);
    }

    @GetMapping("/{userSeq}/bookmarks")
    public ResponseEntity<List<MyMemoResponse>> myBookmarkList(@PathVariable Long userSeq){
        return new ResponseEntity<>(memoService.allBookmarkTrueList(userSeq),HttpStatus.OK);
    }
}
