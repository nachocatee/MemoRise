package com.tjjhtjh.memorise.domain.tag.controller;

import com.tjjhtjh.memorise.domain.memo.exception.MemoException;
import com.tjjhtjh.memorise.domain.tag.service.TaggedUserService;
import com.tjjhtjh.memorise.domain.tag.service.dto.request.TaggedUserRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RequiredArgsConstructor
@RequestMapping("/tag")
@RestController
public class TagController {

    private final TaggedUserService taggedUserService;

    @PostMapping("/user/{memoId}/{userSeq}")
    public ResponseEntity<Object> addUserTag(
            @RequestBody TaggedUserRequest taggedUserRequest,@PathVariable Long userSeq ,@PathVariable Long memoId) throws MemoException {
        taggedUserService.addTagUser(taggedUserRequest,memoId);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @DeleteMapping("/user/{memoId}/{userSeq}")
    public ResponseEntity<Object> deleteUserTag(
            @PathVariable Long userSeq ,@PathVariable Long memoId) throws MemoException {
        taggedUserService.deleteTag(memoId);
        return new ResponseEntity<>(HttpStatus.OK);
    }

}
